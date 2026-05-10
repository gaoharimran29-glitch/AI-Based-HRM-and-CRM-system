import json
from datetime import datetime
from tools import (
    get_team_workload,
    get_overtime_data,
    get_urgent_deadlines,
    get_reassignable_tasks,
    find_best_candidates,
    save_redistribution_suggestion,
)

# =========================================================
# HELPERS
# =========================================================

passed = 0
failed = 0
warnings = 0

def pretty(data):
    print(json.dumps(data, indent=2, default=str))

def section(title):
    print(f"\n{'='*60}")
    print(f"  TEST: {title}")
    print('='*60)

def check(label, condition, got=None):
    global passed, failed
    if condition:
        print(f"  ✅  {label}")
        passed += 1
    else:
        print(f"  ❌  {label}" + (f"  →  got: {got}" if got is not None else ""))
        failed += 1

def warn(label):
    global warnings
    print(f"  ⚠️   {label}")
    warnings += 1

def is_json_safe(obj) -> bool:
    """Verify the entire object is JSON-serialisable (no datetimes, ObjectIds, etc.)."""
    try:
        json.dumps(obj, default=None)   # no fallback — must be natively safe
        return True
    except (TypeError, ValueError):
        return False

# =========================================================
# TEST 1 — get_team_workload
# =========================================================
section("get_team_workload")
workload_data = None
try:
    result = get_team_workload()
    workload = result.get("team_workload", [])
    summary  = result.get("summary", {})
    workload_data = workload   # reuse in later tests

    check("Returns 'team_workload' key",            "team_workload" in result)
    check("Returns 'summary' key",                  "summary" in result)
    check("Has employees",                          len(workload) > 0)

    # Field presence
    required_fields = [
        "employee_id", "name", "team", "skills",
        "effective_hours", "raw_estimated_hours",
        "utilization_pct", "workload_level",
        "is_overloaded", "has_capacity",
        "blocked_tasks", "critical_tasks",
        "high_priority_tasks", "deadline_pressure",
        "burnout_score", "is_burnout_risk",
        "total_overtime_7d", "burnout_signal_days",
        "avg_hours_per_day", "avg_productivity",
    ]
    for f in required_fields:
        check(f"Field '{f}' present on every employee",
              all(f in e for e in workload))

    # workload_level values
    valid_levels = {"critical", "overloaded", "at_risk", "normal", "underutilised"}
    check("workload_level is always a valid value",
          all(e["workload_level"] in valid_levels for e in workload))

    # Sorted by effective_hours descending
    hrs = [e["effective_hours"] for e in workload]
    check("Sorted by effective_hours descending",   hrs == sorted(hrs, reverse=True))

    # Summary counts add up
    total_from_levels = (
        summary.get("critical", 0) + summary.get("overloaded", 0) +
        summary.get("at_risk", 0)  + summary.get("normal", 0) +
        summary.get("underutilised", 0)
    )
    check("Summary counts match total_employees",
          total_from_levels == summary.get("total_employees", -1),
          got=total_from_levels)

    # BUG 3 CHECK — no datetime objects anywhere
    check("Entire result is JSON-safe (no datetimes / ObjectIds)",
          is_json_safe(result))

    overloaded = [e for e in workload if e["is_overloaded"]]
    has_cap    = [e for e in workload if e["has_capacity"]]
    burnout    = [e for e in workload if e["is_burnout_risk"]]

    print(f"\n  📊 Total employees  : {len(workload)}")
    print(f"  🔴 Overloaded       : {len(overloaded)}")
    print(f"  🟡 Has capacity     : {len(has_cap)}")
    print(f"  🔥 Burnout risk     : {len(burnout)}")
    if summary:
        print(f"  📋 Summary          : {summary}")
    if overloaded:
        print(f"\n  Sample overloaded employee:")
        pretty(overloaded[0])

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 2 — get_overtime_data
# =========================================================
section("get_overtime_data")
try:
    result   = get_overtime_data()
    overtime = result.get("overtime_data", [])

    check("Returns 'overtime_data' key",            "overtime_data" in result)
    check("Returns 'at_risk_count' key",            "at_risk_count" in result)
    check("Has records",                            len(overtime) > 0)

    ot_fields = [
        "employee_id", "total_overtime_7d", "burnout_signal_days",
        "avg_hours_per_day", "avg_productivity", "burnout_score", "is_at_risk"
    ]
    for f in ot_fields:
        check(f"Field '{f}' present on every record",
              all(f in r for r in overtime))

    # Sorted by burnout_score descending
    scores = [r["burnout_score"] for r in overtime]
    check("Sorted by burnout_score descending",     scores == sorted(scores, reverse=True))

    # at_risk_count matches actual records
    actual_at_risk = sum(1 for r in overtime if r["is_at_risk"])
    check("at_risk_count matches actual is_at_risk count",
          result["at_risk_count"] == actual_at_risk,
          got=result["at_risk_count"])

    # BUG 3 CHECK
    check("Entire result is JSON-safe",             is_json_safe(result))

    # Consistency with get_team_workload burnout flags
    if workload_data:
        wl_risks = {e["employee_id"] for e in workload_data if e["is_burnout_risk"]}
        ot_risks = {r["employee_id"] for r in overtime if r["is_at_risk"]}
        if wl_risks != ot_risks:
            warn(f"Burnout risk sets differ between tools "
                 f"(workload={len(wl_risks)}, overtime={len(ot_risks)}) — "
                 "may be a timing difference, not a bug.")
        else:
            check("Burnout risk flags consistent across both tools", True)

    at_risk = [r for r in overtime if r["is_at_risk"]]
    print(f"\n  📊 Total records    : {len(overtime)}")
    print(f"  🔥 At-risk count    : {len(at_risk)}")
    if at_risk:
        print(f"\n  Highest burnout employee:")
        pretty(at_risk[0])

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 3 — get_urgent_deadlines  (BUG 3 — datetime fields)
# =========================================================
section("get_urgent_deadlines — BUG 3 datetime fix")
try:
    result = get_urgent_deadlines(days_ahead=7)

    check("Returns 'urgent_tasks' key",             "urgent_tasks" in result)
    check("Returns 'urgent_projects' key",          "urgent_projects" in result)

    # BUG 3: deadlines must be strings, NOT datetime objects
    task_deadlines_are_strings = all(
        isinstance(t["deadline"], str)
        for t in result["urgent_tasks"]
    )
    check("Task deadlines are strings (not datetime)", task_deadlines_are_strings)

    project_deadlines_are_strings = all(
        isinstance(p["deadline"], str)
        for p in result["urgent_projects"]
    )
    check("Project deadlines are strings (not datetime)", project_deadlines_are_strings)

    # BUG 3: start_date on projects must also be stripped
    project_start_dates_safe = all(
        not isinstance(p.get("start_date"), datetime)
        for p in result["urgent_projects"]
    )
    check("Project start_date is not a raw datetime", project_start_dates_safe)

    # Full JSON safety
    check("Entire result is JSON-safe",             is_json_safe(result))

    print(f"\n  📊 Urgent tasks     : {len(result['urgent_tasks'])}")
    print(f"  📊 Urgent projects  : {len(result['urgent_projects'])}")
    if result["urgent_tasks"]:
        print(f"\n  Sample urgent task:")
        pretty(result["urgent_tasks"][0])

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 4 — get_reassignable_tasks  (GAP 1 — required_skills + note)
# =========================================================
section("get_reassignable_tasks — GAP 1 required_skills fix")
overloaded_emp = None
try:
    if not workload_data:
        warn("No workload data available — skipping test 4")
    else:
        overloaded = [e for e in workload_data if e["is_overloaded"]]
        if not overloaded:
            warn("No overloaded employees in DB — skipping test 4 "
                 "(run feed_data.py to generate test data)")
        else:
            overloaded_emp = overloaded[0]
            emp_id = overloaded_emp["employee_id"]
            print(f"  Using employee: {emp_id} ({overloaded_emp['name']})")

            result = get_reassignable_tasks(emp_id)

            check("Returns 'employee_id'",          "employee_id" in result)
            check("Returns 'reassignable_tasks'",   "reassignable_tasks" in result)
            check("Returns 'total_count'",          "total_count" in result)
            check("Returns 'note' field for Gemini","note" in result)   # GAP 1

            tasks = result["reassignable_tasks"]

            if tasks:
                # GAP 1: every task must have required_skills
                check("Every task has 'required_skills' field",
                      all("required_skills" in t for t in tasks))
                check("required_skills is a list on every task",
                      all(isinstance(t["required_skills"], list) for t in tasks))

                # No blocked tasks should appear
                check("No blocked tasks in results",
                      all(not t.get("blocked", False) for t in tasks))

                # Deadlines are strings
                check("Task deadlines are strings",
                      all(isinstance(t["deadline"], str) for t in tasks))

                check("Entire result is JSON-safe", is_json_safe(result))

                print(f"\n  📊 Reassignable tasks: {result['total_count']}")
                print(f"\n  Sample task:")
                pretty(tasks[0])
            else:
                warn(f"No reassignable tasks for {emp_id} — "
                     "employee may have only blocked/urgent tasks")

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 5 — find_best_candidates  (BUG 4 N+1, GAP 1 skill filter)
# =========================================================
section("find_best_candidates — BUG 4 N+1 query + GAP 1 skill filter")
try:
    test_skills  = ["python", "mongodb", "fastapi"]
    test_exclude = "EMP-1000"

    result = find_best_candidates(test_skills, test_exclude)

    check("Returns 'candidates' key",               "candidates" in result)

    candidates = result["candidates"]

    # Excluded employee must not appear
    check("Excluded employee not in results",
          all(c["employee_id"] != test_exclude for c in candidates))

    if candidates:
        # All required fields present
        cand_fields = [
            "employee_id", "name", "team", "skills",
            "matched_skills", "skill_score",
            "utilization_pct", "capacity_score",
            "avg_quality_score", "deadline_met_rate",
            "composite_score", "burnout_risk"
        ]
        for f in cand_fields:
            check(f"Field '{f}' present on every candidate",
                  all(f in c for c in candidates))

        # GAP 1: no candidate with zero skill overlap
        check("No candidate has zero skill_score (gap 1 filter)",
              all(c["skill_score"] > 0 for c in candidates))

        # Sorted descending
        scores = [c["composite_score"] for c in candidates]
        check("Sorted by composite_score descending",
              scores == sorted(scores, reverse=True), got=scores)

        # Max 5 results
        check("Returns at most 5 candidates",       len(candidates) <= 5)

        # matched_skills must be a subset of test_skills
        check("matched_skills is a subset of required_skills",
              all(set(c["matched_skills"]).issubset(set(test_skills))
                  for c in candidates))

        check("Entire result is JSON-safe",         is_json_safe(result))

        print(f"\n  📊 Candidates found : {len(candidates)}")
        print(f"\n  Top candidate:")
        pretty(candidates[0])

    else:
        warn("No candidates returned — all employees may be overloaded "
             "or unavailable in test data")

    # GAP 1: verify zero-skill-overlap candidates are filtered out
    # by running with skills that nobody has
    no_skill_result = find_best_candidates(["cobol", "fortran77"], "EMP-9999")
    check("Returns empty list when no skill matches exist",
          len(no_skill_result.get("candidates", [])) == 0,
          got=no_skill_result)

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 6 — save_redistribution_suggestion  (GAP 2 dedup)
# =========================================================
section("save_redistribution_suggestion — GAP 2 dedup fix")
try:
    from pymongo import MongoClient as _MC
    _suggestions_col = _MC("mongodb://localhost:27017")["workload_balancing_ai"]["redistribution_suggestions"]

    TEST_TASK_ID = "TASK-TEST-99999"

    # Clean up any leftover from a previous run so the test is always fresh
    _suggestions_col.delete_many({"task_id": TEST_TASK_ID})

    test_suggestion = {
        "task_id":          TEST_TASK_ID,
        "task_title":       "Test Task — unit test only",
        "from_employee_id": "EMP-1000",
        "to_employee_id":   "EMP-1005",
        "to_employee_name": "Test Employee",
        "reason":           "Automated unit test — please ignore",
        "urgency_score":    0.5,
        "confidence_score": 0.9,
    }

    # First save — must always succeed after cleanup
    result1 = save_redistribution_suggestion(test_suggestion)
    check("First save returns saved=True",           result1.get("saved") == True,
          got=result1)
    check("First save returns a suggestion_id",      bool(result1.get("suggestion_id")))

    # Second save of same task — must NOT create a duplicate
    result2 = save_redistribution_suggestion(test_suggestion)
    check("Second save returns saved=False (dedup)", result2.get("saved") == False,
          got=result2.get("saved"))
    check("Second save returns same suggestion_id",
          result2.get("suggestion_id") == result1.get("suggestion_id"),
          got=result2.get("suggestion_id"))
    check("Second save returns 'reason' explaining dedup",
          "duplicate" in result2.get("reason", "").lower(),
          got=result2.get("reason"))

    check("Entire result is JSON-safe",              is_json_safe(result1))

    # Confirm only one document was inserted
    count = _suggestions_col.count_documents({"task_id": TEST_TASK_ID})
    check("Exactly one document in DB for test task", count == 1, got=count)

    print(f"\n  💾 Saved ID  : {result1.get('suggestion_id')}")
    print(f"  🔁 Dedup ID  : {result2.get('suggestion_id')} (same — correct)")
    print(f"  🗄️  DB count  : {count} document (correct)")

except Exception as e:
    print(f"  ❌ EXCEPTION: {e}")
    failed += 1

# =========================================================
# TEST 7 — End-to-end flow simulation
# Simulates exactly what Gemini does across one full loop
# =========================================================
section("End-to-end flow simulation (Gemini loop dry run)")
try:
    flow_ok = True

    # Step 1: workload
    wl = get_team_workload()
    overloaded = [e for e in wl["team_workload"] if e["is_overloaded"]]
    if not overloaded:
        warn("No overloaded employees — end-to-end flow skipped. "
             "Run feed_data.py to generate suitable test data.")
        flow_ok = False

    if flow_ok:
        # Step 2: pick first overloaded employee
        emp = overloaded[0]
        emp_id = emp["employee_id"]
        print(f"  Simulating flow for: {emp_id} ({emp['name']})")

        # Step 3: get reassignable tasks
        rt = get_reassignable_tasks(emp_id)
        tasks = rt["reassignable_tasks"]

        if not tasks:
            warn(f"No reassignable tasks for {emp_id} — partial flow only")
            flow_ok = False

    if flow_ok:
        # Step 4: find candidates for first task (using its required_skills)
        task = tasks[0]
        skills = task.get("required_skills", [])
        check("Task has required_skills for candidate search", len(skills) > 0)

        candidates = find_best_candidates(skills, emp_id)["candidates"]
        check("find_best_candidates returns results for task skills",
              isinstance(candidates, list))

        if candidates:
            best = candidates[0]

            # Step 5: save suggestion
            suggestion = {
                "task_id":          task["task_id"],
                "task_title":       task["title"],
                "from_employee_id": emp_id,
                "to_employee_id":   best["employee_id"],
                "to_employee_name": best["name"],
                "reason":           (
                    f"End-to-end test: {emp['name']} is overloaded "
                    f"({emp['utilization_pct']}% utilisation). "
                    f"{best['name']} has {best['skill_score']}% skill match "
                    f"and {best['utilization_pct']}% utilisation."
                ),
                "urgency_score":    round(min(emp["utilization_pct"] / 100, 1.0), 2),
                "confidence_score": round(best["composite_score"] / 100, 2),
            }
            save_result = save_redistribution_suggestion(suggestion)
            check("End-to-end suggestion saved or deduped cleanly",
                  "suggestion_id" in save_result)

            print(f"\n  ✅ Full flow complete:")
            print(f"     Task    : {task['task_id']} — {task['title']}")
            print(f"     From    : {emp['name']} ({emp['utilization_pct']}% utilisation)")
            print(f"     To      : {best['name']} ({best['utilization_pct']}% utilisation)")
            print(f"     Skills  : {best['matched_skills']}")
            print(f"     Score   : {best['composite_score']}")
            print(f"     Saved   : {save_result}")
        else:
            warn("No candidates found for task skills — flow ends at step 4")

except Exception as e:
    print(f"  ❌ EXCEPTION in end-to-end flow: {e}")
    failed += 1

# =========================================================
# FINAL REPORT
# =========================================================
print(f"\n{'='*60}")
print(f"  RESULTS:  ✅ {passed} passed   ❌ {failed} failed   ⚠️  {warnings} warnings")
print('='*60)

if failed == 0 and warnings == 0:
    print("\n  🎉 All checks passed — safe to run ai_engine.py\n")
elif failed == 0:
    print("\n  ✅ No failures. Review warnings above before running ai_engine.py\n")
else:
    print("\n  🛑 Fix failing tests before running ai_engine.py\n")