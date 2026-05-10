import os
import random
import uuid
from datetime import datetime, timedelta

from faker import Faker
from pymongo import MongoClient, ASCENDING

fake = Faker()

# =========================================================
# CONNECTION  (BUG-1)
# =========================================================

MONGO_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME   = os.getenv("DB_NAME",    "workload_balancing_ai")

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

employees_col          = db["employees"]
projects_col           = db["projects"]
tasks_col              = db["tasks"]
timesheets_col         = db["timesheets"]
assignment_history_col = db["assignment_history"]
suggestions_col        = db["redistribution_suggestions"]

# =========================================================
# CONSTANTS
# =========================================================

ALL_SKILLS = [
    "python", "fastapi", "django", "mongodb", "postgresql", "redis",
    "docker", "kubernetes", "aws", "machine learning", "deep learning",
    "pandas", "numpy", "react", "nodejs", "typescript", "java",
    "spring boot", "system design", "microservices", "devops",
    "data engineering", "airflow", "spark", "nlp", "tensorflow",
    "pytorch", "backend", "frontend", "api development", "async programming"
]

TEAMS             = ["backend", "frontend", "ml", "devops", "data-engineering"]
TASK_PRIORITIES   = ["low", "medium", "high", "critical"]
TASK_STATUSES     = ["todo", "in_progress", "blocked", "review", "completed"]
PROJECT_STATUSES  = ["active", "planning", "at_risk", "completed"]

PROJECT_NAMES = [
    "AI Hiring Platform", "Enterprise CRM", "Fraud Detection Engine",
    "Cloud Migration", "Recommendation System", "Analytics Dashboard",
    "Customer Support AI", "Data Lake Modernization",
    "Workflow Automation System", "AI Productivity Assistant",
]

TASK_TITLES = [
    "Build authentication API", "Optimize MongoDB queries",
    "Implement caching layer", "Train ML recommendation model",
    "Develop analytics dashboard", "Refactor async services",
    "Migrate legacy APIs", "Implement CI/CD pipeline",
    "Create monitoring system", "Design microservice architecture",
    "Implement vector search", "Improve API performance",
    "Create NLP pipeline", "Deploy Kubernetes cluster",
    "Build workflow automation",
]

# =========================================================
# SEED  (BUG-2 FIX: only runs as __main__)
# =========================================================

def seed():
    print("Clearing existing data…")
    for col in [employees_col, projects_col, tasks_col,
                timesheets_col, assignment_history_col, suggestions_col]:
        col.delete_many({})

    # ── Employees ──────────────────────────────────────────
    employees = []
    for i in range(40):
        employees.append({
            "employee_id":             f"EMP-{1000+i}",
            "name":                    fake.name(),
            "email":                   fake.email(),
            "team":                    random.choice(TEAMS),
            "designation":             random.choice([
                "Software Engineer", "Senior Software Engineer",
                "ML Engineer", "DevOps Engineer", "Data Engineer", "Tech Lead"
            ]),
            "experience_years":        random.randint(1, 12),
            "skills":                  random.sample(ALL_SKILLS, random.randint(5, 10)),
            "capacity_hours_per_week": 40,
            "max_safe_hours_per_week": 45,
            "timezone":                "Asia/Kolkata",
            "availability_status":     random.choice(
                ["available", "busy", "partially_available"]
            ),
            "performance_score":       round(random.uniform(65, 98), 2),
            "burnout_risk_score":      round(random.uniform(0.1, 0.9), 2),
            "created_at":              datetime.utcnow(),
            "updated_at":              datetime.utcnow(),
        })
    employees_col.insert_many(employees)
    print(f"Inserted {len(employees)} employees")

    # ── Projects ───────────────────────────────────────────
    projects = []
    for i in range(10):
        start = datetime.utcnow() - timedelta(days=random.randint(10, 90))
        projects.append({
            "project_id":            f"PROJ-{500+i}",
            "project_name":          random.choice(PROJECT_NAMES),
            "description":           fake.text(max_nb_chars=200),
            "priority":              random.choice(TASK_PRIORITIES),
            "status":                random.choice(PROJECT_STATUSES),
            "start_date":            start,
            "deadline":              datetime.utcnow() + timedelta(days=random.randint(15, 120)),
            "budget":                random.randint(100_000, 5_000_000),
            "risk_score":            round(random.uniform(0.1, 0.95), 2),
            "completion_percentage": random.randint(5, 95),
            "team_size":             random.randint(4, 15),
            "created_at":            datetime.utcnow(),
        })
    projects_col.insert_many(projects)
    print(f"Inserted {len(projects)} projects")

    # ── Tasks ──────────────────────────────────────────────
    employees_data = list(employees_col.find())
    projects_data  = list(projects_col.find())
    tasks = []
    for i in range(250):
        emp     = random.choice(employees_data)
        project = random.choice(projects_data)
        est_hrs = random.randint(4, 40)
        tasks.append({
            "task_id":               f"TASK-{10000+i}",
            "title":                 random.choice(TASK_TITLES),
            "description":           fake.text(max_nb_chars=400),
            "project_id":            project["project_id"],
            "assigned_to":           emp["employee_id"],
            "priority":              random.choice(TASK_PRIORITIES),
            "status":                random.choice(TASK_STATUSES),
            "estimated_hours":       est_hrs,
            "actual_hours":          max(est_hrs + random.randint(-2, 15), 1),
            "complexity_score":      round(random.uniform(1, 10), 2),
            "required_skills":       random.sample(ALL_SKILLS, random.randint(2, 5)),
            "dependency_count":      random.randint(0, 5),
            "deadline":              datetime.utcnow() + timedelta(days=random.randint(1, 45)),
            "created_at":            datetime.utcnow() - timedelta(days=random.randint(1, 60)),
            "updated_at":            datetime.utcnow(),
            "completion_percentage": random.randint(0, 100),
            "blocked":               random.choice([True, False]),
            "ai_risk_score":         round(random.uniform(0.1, 0.95), 2),
        })
    tasks_col.insert_many(tasks)
    print(f"Inserted {len(tasks)} tasks")

    # ── Timesheets  (BUG-3 FIX: burnout_signal correlated with hours) ──
    timesheets = []
    for emp in employees_data:
        for day in range(30):
            hours_worked  = round(random.uniform(4, 12), 2)
            overtime_hrs  = max(hours_worked - 8, 0)
            # burnout signal is True when overtime > 1h AND random chance
            burnout_signal = overtime_hrs > 1.0 and random.random() < 0.6
            timesheets.append({
                "timesheet_id":      str(uuid.uuid4()),
                "employee_id":       emp["employee_id"],
                "date":              datetime.utcnow() - timedelta(days=day),
                "hours_worked":      hours_worked,
                "overtime_hours":    overtime_hrs,
                "productivity_score": round(
                    # productivity drops when hours are high
                    random.uniform(60, 100) * max(1 - (overtime_hrs / 20), 0.7), 2
                ),
                "meetings_count":    random.randint(0, 8),
                "tasks_completed":   random.randint(0, 5),
                "focus_time_hours":  round(random.uniform(1, 7), 2),
                "burnout_signal":    burnout_signal,
            })
    timesheets_col.insert_many(timesheets)
    print(f"Inserted {len(timesheets)} timesheets")

    # ── Assignment history ─────────────────────────────────
    history = []
    for i in range(300):
        emp = random.choice(employees_data)
        history.append({
            "history_id":           str(uuid.uuid4()),
            "employee_id":          emp["employee_id"],
            "task_category":        random.choice(
                ["backend", "frontend", "ml", "devops", "database", "optimization"]
            ),
            "completion_time_hours": random.randint(2, 60),
            "quality_score":        round(random.uniform(60, 100), 2),
            "deadline_met":         random.choice([True, False]),
            "reassigned":           random.choice([True, False]),
            "created_at":           datetime.utcnow() - timedelta(days=random.randint(1, 365)),
        })
    assignment_history_col.insert_many(history)
    print(f"Inserted {len(history)} assignment history records")

    # ── Indexes  (BUG-4 FIX: compound index on timesheets) ─
    employees_col.create_index([("employee_id", ASCENDING)], unique=True)
    employees_col.create_index([("skills", ASCENDING)])
    employees_col.create_index([("team", ASCENDING)])

    projects_col.create_index([("project_id", ASCENDING)], unique=True)
    projects_col.create_index([("deadline", ASCENDING)])

    tasks_col.create_index([("task_id", ASCENDING)], unique=True)
    tasks_col.create_index([("assigned_to", ASCENDING)])
    tasks_col.create_index([("priority", ASCENDING)])
    tasks_col.create_index([("deadline", ASCENDING)])
    tasks_col.create_index([("required_skills", ASCENDING)])

    timesheets_col.create_index([("employee_id", ASCENDING)])
    timesheets_col.create_index([("date", ASCENDING)])
    timesheets_col.create_index([("employee_id", ASCENDING), ("date", ASCENDING)])

    assignment_history_col.create_index([("employee_id", ASCENDING)])

    suggestions_col.create_index([("task_id", ASCENDING), ("status", ASCENDING)])

    print("Indexes created successfully.")
    print("\nDataset seeded successfully.")


if __name__ == "__main__":
    seed()
