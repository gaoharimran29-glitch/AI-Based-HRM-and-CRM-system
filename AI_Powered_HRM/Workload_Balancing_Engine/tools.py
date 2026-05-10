import logging
import os
from datetime import datetime, timedelta

from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

log = logging.getLogger(__name__)

# =========================================================
# DB CONNECTION  (BUG-1, BUG-2)
# =========================================================

_MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
_DB_NAME   = os.getenv("DB_NAME",    "workload_balancing_ai")

_mongo_client = None

def _get_db():
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(
            _MONGO_URI,
            serverSelectionTimeoutMS=5_000,
            connectTimeoutMS=5_000,
            socketTimeoutMS=10_000,
            maxPoolSize=10,
        )
    return _mongo_client[_DB_NAME]

def _employees():          return _get_db()["employees"]
def _tasks():              return _get_db()["tasks"]
def _timesheets():         return _get_db()["timesheets"]
def _projects():           return _get_db()["projects"]
def _assignment_history(): return _get_db()["assignment_history"]
def _suggestions():        return _get_db()["redistribution_suggestions"]

# =========================================================
# INDEXES  (GAP-3)
# =========================================================

def ensure_indexes():
    try:
        _employees().create_index([("employee_id", ASCENDING)], unique=True)
        _employees().create_index([("skills", ASCENDING)])
        _tasks().create_index([("task_id", ASCENDING)], unique=True)
        _tasks().create_index([("assigned_to", ASCENDING)])
        _tasks().create_index([("deadline", ASCENDING)])
        _tasks().create_index([("status", ASCENDING)])
        _tasks().create_index([("required_skills", ASCENDING)])
        _projects().create_index([("project_id", ASCENDING)], unique=True)
        _projects().create_index([("deadline", ASCENDING)])
        _timesheets().create_index([("employee_id", ASCENDING)])
        _timesheets().create_index([("employee_id", ASCENDING), ("date", ASCENDING)])
        _assignment_history().create_index([("employee_id", ASCENDING)])
        _suggestions().create_index([("task_id", ASCENDING), ("status", ASCENDING)])
        log.info("MongoDB indexes verified.")
    except PyMongoError as exc:
        log.warning("Index creation failed (non-fatal): %s", exc)

# =========================================================
# CONSTANTS
# =========================================================

PRIORITY_HOUR_WEIGHT = {"low": 1.0, "medium": 1.3, "high": 1.6, "critical": 2.0}
BLOCKED_HOUR_PENALTY  = 0.25
OVERLOAD_THRESHOLD    = 1.10
RISK_THRESHOLD        = 0.90
CAPACITY_THRESHOLD    = 0.70
BURNOUT_OVERTIME_7D   = 8
BURNOUT_SIGNAL_DAYS   = 3
BURNOUT_AVG_HOURS_DAY = 9.5

# =========================================================
# RETRY DECORATOR  (GAP-5)
# =========================================================

_mongo_retry = retry(
    retry=retry_if_exception_type(PyMongoError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)

# =========================================================
# HELPERS
# =========================================================

def _utcnow():
    return datetime.utcnow()

def _effective_hours(task):
    raw        = task.get("estimated_hours") or 0
    priority   = task.get("priority", "medium")
    complexity = task.get("complexity_score") or 5.0
    completion = (task.get("completion_percentage") or 0) / 100.0
    blocked    = task.get("blocked", False)
    remaining  = raw * (1.0 - completion)
    if blocked:
        return remaining * BLOCKED_HOUR_PENALTY
    p_w = PRIORITY_HOUR_WEIGHT.get(priority, 1.3)
    c_w = 0.8 + (complexity - 1) * (0.6 / 9)
    return remaining * p_w * c_w

def _strip_datetimes(obj):
    """BUG-3 FIX: recursively convert datetime → ISO string."""
    if isinstance(obj, dict):
        return {k: _strip_datetimes(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_datetimes(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.strftime("%Y-%m-%dT%H:%M:%S")
    return obj

@_mongo_retry
def _compute_burnout_score(employee_id, seven_days_ago):
    pipeline = [
        {"$match": {"employee_id": employee_id, "date": {"$gte": seven_days_ago}}},
        {"$group": {
            "_id": None,
            "total_hours":      {"$sum": "$hours_worked"},
            "total_overtime":   {"$sum": "$overtime_hours"},
            "burnout_days":     {"$sum": {"$cond": ["$burnout_signal", 1, 0]}},
            "avg_productivity": {"$avg": "$productivity_score"},
            "day_count":        {"$sum": 1}
        }}
    ]
    rows = list(_timesheets().aggregate(pipeline))
    if not rows:
        return {"burnout_score": 0.3, "total_overtime_7d": 0,
                "burnout_signal_days": 0, "avg_hours_per_day": 0.0,
                "avg_productivity": 75.0, "is_burnout_risk": False}
    r           = rows[0]
    day_count   = max(r["day_count"], 1)
    avg_hrs_day = round(r["total_hours"] / day_count, 2)
    overtime_s  = min(r["total_overtime"] / BURNOUT_OVERTIME_7D, 1.0)
    signal_s    = min(r["burnout_days"]   / BURNOUT_SIGNAL_DAYS,  1.0)
    hours_s     = min(max(avg_hrs_day - 8, 0) / (BURNOUT_AVG_HOURS_DAY - 8), 1.0)
    prod_inv    = max(0, (80 - (r["avg_productivity"] or 80)) / 80)
    score       = overtime_s * 0.35 + signal_s * 0.30 + hours_s * 0.20 + prod_inv * 0.15
    return {
        "burnout_score":       round(min(score, 1.0), 3),
        "total_overtime_7d":   round(r["total_overtime"], 2),
        "burnout_signal_days": r["burnout_days"],
        "avg_hours_per_day":   avg_hrs_day,
        "avg_productivity":    round(r["avg_productivity"] or 0, 2),
        "is_burnout_risk": (
            r["total_overtime"] >= BURNOUT_OVERTIME_7D or
            r["burnout_days"]   >= BURNOUT_SIGNAL_DAYS or
            avg_hrs_day         >= BURNOUT_AVG_HOURS_DAY
        )
    }

# =========================================================
# TOOL 1 — Team workload
# =========================================================

@_mongo_retry
def get_team_workload():
    now            = _utcnow()
    seven_days_ago = now - timedelta(days=7)
    deadline_72h   = now + timedelta(hours=72)

    active_tasks = list(_tasks().find(
        {"status": {"$in": ["todo", "in_progress", "blocked"]}},
        {"_id": 0, "task_id": 1, "assigned_to": 1, "estimated_hours": 1,
         "priority": 1, "complexity_score": 1, "completion_percentage": 1,
         "blocked": 1, "deadline": 1}
    ))
    tasks_by_emp = {}
    for t in active_tasks:
        tasks_by_emp.setdefault(t["assigned_to"], []).append(t)

    employees = list(_employees().find(
        {},
        {"_id": 0, "employee_id": 1, "name": 1, "team": 1,
         "capacity_hours_per_week": 1, "availability_status": 1, "skills": 1}
    ))

    result = []
    for emp in employees:
        eid      = emp["employee_id"]
        capacity = emp.get("capacity_hours_per_week") or 40   # BUG-6 FIX
        emp_tasks = tasks_by_emp.get(eid, [])

        effective_hrs = sum(_effective_hours(t) for t in emp_tasks)
        raw_hrs       = sum((t.get("estimated_hours") or 0) for t in emp_tasks)
        utilization   = effective_hrs / capacity

        blocked_count     = sum(1 for t in emp_tasks if t.get("blocked"))
        critical_count    = sum(1 for t in emp_tasks if t.get("priority") == "critical")
        high_count        = sum(1 for t in emp_tasks if t.get("priority") == "high")
        deadline_pressure = sum(
            1 for t in emp_tasks
            if t.get("deadline") and t["deadline"] <= deadline_72h and not t.get("blocked")
        )

        if utilization >= OVERLOAD_THRESHOLD:
            level = "critical" if critical_count >= 2 else "overloaded"
        elif utilization >= RISK_THRESHOLD:
            level = "at_risk"
        elif utilization < CAPACITY_THRESHOLD:
            level = "underutilised"
        else:
            level = "normal"

        burnout = _compute_burnout_score(eid, seven_days_ago)

        result.append({
            "employee_id":         eid,
            "name":                emp["name"],
            "team":                emp["team"],
            "skills":              emp.get("skills", []),
            "availability_status": emp.get("availability_status", "unknown"),
            "active_task_count":   len(emp_tasks),
            "raw_estimated_hours": raw_hrs,
            "effective_hours":     round(effective_hrs, 2),
            "capacity_hours":      capacity,
            "utilization_pct":     round(utilization * 100, 1),
            "workload_level":      level,
            "is_overloaded":       utilization >= OVERLOAD_THRESHOLD,
            "has_capacity":        utilization < CAPACITY_THRESHOLD,
            "blocked_tasks":       blocked_count,
            "critical_tasks":      critical_count,
            "high_priority_tasks": high_count,
            "deadline_pressure":   deadline_pressure,
            **burnout
        })

    result.sort(key=lambda e: e["effective_hours"], reverse=True)
    return _strip_datetimes({
        "team_workload": result,
        "summary": {
            "total_employees": len(result),
            "critical":        sum(1 for e in result if e["workload_level"] == "critical"),
            "overloaded":      sum(1 for e in result if e["workload_level"] == "overloaded"),
            "at_risk":         sum(1 for e in result if e["workload_level"] == "at_risk"),
            "normal":          sum(1 for e in result if e["workload_level"] == "normal"),
            "underutilised":   sum(1 for e in result if e["workload_level"] == "underutilised"),
            "burnout_risks":   sum(1 for e in result if e["is_burnout_risk"]),
        }
    })

# =========================================================
# TOOL 2 — Overtime / burnout
# =========================================================

@_mongo_retry
def get_overtime_data():
    seven_days_ago = _utcnow() - timedelta(days=7)
    all_emp_ids = [
        e["employee_id"]
        for e in _employees().find({}, {"_id": 0, "employee_id": 1})
    ]
    report = []
    for eid in all_emp_ids:
        b = _compute_burnout_score(eid, seven_days_ago)
        report.append({
            "employee_id":         eid,
            "total_overtime_7d":   b["total_overtime_7d"],
            "burnout_signal_days": b["burnout_signal_days"],
            "avg_hours_per_day":   b["avg_hours_per_day"],
            "avg_productivity":    b["avg_productivity"],
            "burnout_score":       b["burnout_score"],
            "is_at_risk":          b["is_burnout_risk"],
        })
    report.sort(key=lambda r: r["burnout_score"], reverse=True)
    return _strip_datetimes({
        "overtime_data": report,
        "at_risk_count": sum(1 for r in report if r["is_at_risk"])
    })

# =========================================================
# TOOL 3 — Urgent deadlines
# =========================================================

@_mongo_retry
def get_urgent_deadlines(days_ahead=7):
    now    = _utcnow()
    cutoff = now + timedelta(days=days_ahead)

    urgent_tasks = list(_tasks().find(
        {"deadline":              {"$exists": True, "$lte": cutoff},
         "status":                {"$nin": ["completed"]},
         "completion_percentage": {"$lt": 100}},
        {"_id": 0, "task_id": 1, "title": 1, "assigned_to": 1,
         "priority": 1, "deadline": 1, "status": 1,
         "completion_percentage": 1, "required_skills": 1,
         "estimated_hours": 1, "blocked": 1}
    ).sort("deadline", ASCENDING))

    urgent_projects = list(_projects().find(
        {"deadline":              {"$exists": True, "$lte": cutoff},
         "status":                {"$nin": ["completed"]},
         "completion_percentage": {"$lt": 100}},
        {"_id": 0, "project_id": 1, "project_name": 1, "deadline": 1,
         "start_date": 1, "status": 1, "priority": 1,
         "completion_percentage": 1, "risk_score": 1}
    ).sort("deadline", ASCENDING))

    # BUG-3 FIX: _strip_datetimes handles all nested datetime fields
    return _strip_datetimes({"urgent_tasks": urgent_tasks, "urgent_projects": urgent_projects})

# =========================================================
# TOOL 4 — Reassignable tasks
# =========================================================

@_mongo_retry
def get_reassignable_tasks(employee_id):
    if not employee_id or not isinstance(employee_id, str):
        return {"error": "employee_id must be a non-empty string"}

    tomorrow = _utcnow() + timedelta(days=1)

    tasks = list(_tasks().find(
        {"assigned_to": employee_id,
         "status":      {"$in": ["todo", "in_progress"]},
         "blocked":     False,
         "deadline":    {"$exists": True, "$gt": tomorrow}},   # BUG-5 FIX
        {"_id": 0, "task_id": 1, "title": 1, "priority": 1,
         "estimated_hours": 1, "deadline": 1,
         "required_skills": 1,
         "complexity_score": 1, "completion_percentage": 1}
    ).sort("priority", ASCENDING))

    return _strip_datetimes({
        "employee_id":        employee_id,
        "reassignable_tasks": tasks,
        "total_count":        len(tasks),
        "note": (
            "For each task, call find_best_candidates with that task's "
            "required_skills list and exclude_employee_id = this employee_id."
        )
    })

# =========================================================
# TOOL 5 — Find best candidates
# =========================================================

@_mongo_retry
def find_best_candidates(required_skills, exclude_employee_id, _workload_snapshot=None):
    if not isinstance(required_skills, list):
        return {"error": "required_skills must be a list"}
    if not exclude_employee_id:
        return {"error": "exclude_employee_id is required"}

    # BUG-7 FIX: use snapshot when available to avoid redundant DB scan
    workload_data = _workload_snapshot or get_team_workload()["team_workload"]

    available_emps = {
        e["employee_id"]: e
        for e in workload_data
        if e["employee_id"] != exclude_employee_id
        and not e["is_overloaded"]
        and e.get("availability_status") in ("available", "partially_available")
    }

    if not available_emps:
        return {"candidates": [], "message": "No available candidates found"}

    # BUG-4 FIX — single aggregation for all candidates
    history_pipeline = [
        {"$match": {"employee_id": {"$in": list(available_emps.keys())}}},
        {"$group": {
            "_id":           "$employee_id",
            "scores":        {"$push": "$quality_score"},
            "deadlines_met": {"$sum": {"$cond": ["$deadline_met", 1, 0]}},
            "total":         {"$sum": 1}
        }}
    ]
    history_scores = {}
    for row in _assignment_history().aggregate(history_pipeline):
        history_scores[row["_id"]] = row

    required_set = set(required_skills)
    candidates   = []

    for eid, emp in available_emps.items():
        matched     = set(emp.get("skills", [])) & required_set
        skill_score = (len(matched) / len(required_set)) * 100 if required_set else 0

        # GAP-1 FIX
        if skill_score == 0:
            continue

        capacity_score = max(0.0, 100 - emp["utilization_pct"])
        hist           = history_scores.get(eid, {})
        avg_quality    = sum(hist["scores"]) / len(hist["scores"]) if hist.get("scores") else 70.0
        deadline_rate  = hist["deadlines_met"] / hist["total"] * 100 if hist.get("total") else 70.0
        composite = (
            skill_score    * 0.40 +
            capacity_score * 0.30 +
            avg_quality    * 0.20 +
            deadline_rate  * 0.10
        )
        candidates.append({
            "employee_id":       eid,
            "name":              emp["name"],
            "team":              emp["team"],
            "skills":            emp.get("skills", []),
            "matched_skills":    list(matched),
            "skill_score":       round(skill_score, 1),
            "utilization_pct":   emp["utilization_pct"],
            "capacity_score":    round(capacity_score, 1),
            "avg_quality_score": round(avg_quality, 1),
            "deadline_met_rate": round(deadline_rate, 1),
            "composite_score":   round(composite, 1),
            "burnout_risk":      emp.get("burnout_score", 0),
        })

    candidates.sort(key=lambda x: x["composite_score"], reverse=True)
    return {"candidates": candidates[:5]}

# =========================================================
# TOOL 6 — Save suggestion
# =========================================================

_SUGGESTION_REQUIRED = [
    "task_id", "task_title", "from_employee_id",
    "to_employee_id", "to_employee_name", "reason",
    "urgency_score", "confidence_score",
]

@_mongo_retry
def save_redistribution_suggestion(suggestion):
    # BUG-8 FIX — validate required fields
    missing = [f for f in _SUGGESTION_REQUIRED if not suggestion.get(f)]
    if missing:
        return {"saved": False, "error": f"Missing required fields: {missing}"}

    for score_field in ("urgency_score", "confidence_score"):
        val = suggestion.get(score_field)
        if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
            return {"saved": False, "error": f"{score_field} must be a float 0.0–1.0, got {val!r}"}

    # GAP-2 FIX — dedup
    existing = _suggestions().find_one({"task_id": suggestion["task_id"], "status": "pending"})
    if existing:
        return {"saved": False, "reason": "duplicate — pending suggestion already exists",
                "suggestion_id": str(existing["_id"])}

    doc = {**{k: v for k, v in suggestion.items() if k != "_id"},
           "status": "pending", "created_at": _utcnow(), "acted_on": False}
    result = _suggestions().insert_one(doc)
    return {"saved": True, "suggestion_id": str(result.inserted_id)}

# =========================================================
# TOOL DEFINITIONS
# =========================================================

TOOLS = [
    {
        "name": "get_team_workload",
        "description": (
            "Get the current workload summary for all team members. "
            "Returns task counts, effective (weighted) hours, overload flags, "
            "burnout risk scores, and capacity utilisation."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_overtime_data",
        "description": "Get overtime and burnout data for all employees for the last 7 days.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "get_urgent_deadlines",
        "description": (
            "Get tasks and projects with upcoming deadlines. "
            "Use days_ahead to control how far ahead to look (default 7)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "description": "Days ahead to check (default 7)"}
            },
            "required": []
        }
    },
    {
        "name": "get_reassignable_tasks",
        "description": (
            "Get tasks for an overloaded employee that are safe to reassign. "
            "Excludes blocked tasks and tasks due within 24 hours. "
            "Each task includes required_skills — pass directly to find_best_candidates."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string", "description": "Employee ID of overloaded person"}
            },
            "required": ["employee_id"]
        }
    },
    {
        "name": "find_best_candidates",
        "description": (
            "Find the best available employees to take a task, "
            "ranked by skill match, capacity, and past performance. "
            "Employees with zero skill overlap are excluded automatically."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "required_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Skills required for the task"
                },
                "exclude_employee_id": {
                    "type": "string",
                    "description": "Employee ID to exclude (current assignee)"
                }
            },
            "required": ["required_skills", "exclude_employee_id"]
        }
    },
    {
        "name": "save_redistribution_suggestion",
        "description": (
            "Save a redistribution suggestion. "
            "Skips duplicates — if a pending suggestion exists for this task_id, "
            "returns the existing one instead."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":          {"type": "string"},
                "task_title":       {"type": "string"},
                "from_employee_id": {"type": "string"},
                "to_employee_id":   {"type": "string"},
                "to_employee_name": {"type": "string"},
                "reason":           {"type": "string"},
                "urgency_score":    {"type": "number", "description": "0.0–1.0"},
                "confidence_score": {"type": "number", "description": "0.0–1.0"}
            },
            "required": [
                "task_id", "task_title", "from_employee_id",
                "to_employee_id", "to_employee_name",
                "reason", "urgency_score", "confidence_score"
            ]
        }
    }
]

# =========================================================
# TOOL DISPATCHER  (GAP-4 FIX)
# =========================================================

def run_tool(tool_name, tool_input):
    try:
        if tool_name == "get_team_workload":
            return get_team_workload()
        elif tool_name == "get_overtime_data":
            return get_overtime_data()
        elif tool_name == "get_urgent_deadlines":
            return get_urgent_deadlines(days_ahead=tool_input.get("days_ahead", 7))
        elif tool_name == "get_reassignable_tasks":
            return get_reassignable_tasks(tool_input["employee_id"])
        elif tool_name == "find_best_candidates":
            return find_best_candidates(
                required_skills=tool_input["required_skills"],
                exclude_employee_id=tool_input["exclude_employee_id"]
            )
        elif tool_name == "save_redistribution_suggestion":
            return save_redistribution_suggestion(tool_input)
        else:
            log.warning("Unknown tool requested: %s", tool_name)
            return {"error": f"Unknown tool: {tool_name}"}
    except PyMongoError as exc:
        log.error("DB error in tool %s: %s", tool_name, exc)
        return {"error": f"Database error: {exc}"}
    except Exception as exc:
        log.exception("Unexpected error in tool %s", tool_name)
        return {"error": f"Unexpected error: {exc}"}
