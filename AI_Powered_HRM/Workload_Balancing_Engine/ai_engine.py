import json
import logging
import os
import time

from dotenv import load_dotenv
from pathlib import Path

from google import genai
from google.genai import types

from tools import TOOLS, run_tool, ensure_indexes

log = logging.getLogger(__name__)

# =========================================================
# LOAD .ENV
# =========================================================

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path)

# =========================================================
# STARTUP VALIDATION
# =========================================================

_API_KEY = os.getenv("GEMINI_API_KEY")

if not _API_KEY:
    raise EnvironmentError(
        "GEMINI_API_KEY is not set. "
        "Copy .env.example to .env and add your key."
    )

client = genai.Client(api_key=_API_KEY)

MODEL_NAME = "gemini-2.5-flash-lite"

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are an intelligent Workload Balancing Engine for a software engineering company.

Your responsibilities:

1. Analyse team workload using get_team_workload.
2. Detect overloaded employees (is_overloaded = true) and burnout risks (is_burnout_risk = true).
3. For EACH overloaded employee, call get_reassignable_tasks with their employee_id.
4. For EACH reassignable task, call find_best_candidates — always pass the task's
   exact required_skills array and the current assignee as exclude_employee_id.
5. Call save_redistribution_suggestion for every strong match found.
6. Give a final summary (see format below).

Rules:
- Never move blocked tasks.
- Never move tasks due within 24 hours.
- Prefer employees with has_capacity = true.
- Prefer strong skill matches (skill_score > 50).
- Explain every recommendation clearly in the reason field.
- Be proactive and decisive — do not skip employees or tasks.

FINAL SUMMARY FORMAT — always follow this structure exactly:

=== WORKLOAD BALANCING REPORT ===

TEAM SNAPSHOT
  Total employees : <n>
  Overloaded      : <n>  |  At-risk : <n>  |  Normal : <n>  |  Underutilised : <n>
  Burnout risks   : <n>

OVERLOADED EMPLOYEES
For each overloaded employee, one line per employee:
  • <Name> (<ID>) | Utilization: <utilization_pct>% | Effective hrs: <effective_hours>h
      Burnout score: <burnout_score> | Overtime (7d): <total_overtime_7d>h
      Avg hrs/day: <avg_hours_per_day>h | Burnout signal days: <burnout_signal_days>
      Active tasks: <active_task_count>  Critical: <critical_tasks>  Blocked: <blocked_tasks>

BURNOUT RISKS (not already overloaded)
For each burnout-risk employee who is NOT overloaded:
  • <Name> (<ID>) | Burnout score: <burnout_score> | Overtime (7d): <total_overtime_7d>h
      Avg hrs/day: <avg_hours_per_day>h | Signal days: <burnout_signal_days>

TASK REDISTRIBUTIONS
For each saved suggestion:
  Task : <task_id> — <task_title>
  From : <from_employee_name> (<from_employee_id>)
  To   : <to_employee_name> (<to_employee_id>)
  Candidate fit  : Skill score <skill_score>%  |  Composite score <composite_score>
  Assignee load  : <utilization_pct>% utilization  |  Burnout risk <burnout_risk>
  Reason : <reason>

OVERALL SUMMARY
  Tasks redistributed  : <n>
  Employees relieved   : <list of names>
  Key actions          : <2-3 sentence plain-English conclusion>
"""

# =========================================================
# HELPERS
# =========================================================

def _convert_tools_for_gemini():
    declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t["description"],
            parameters=t["input_schema"],
        )
        for t in TOOLS
    ]
    return [types.Tool(function_declarations=declarations)]


def _safe_text(response) -> str:
    """BUG-3 FIX: extract text without crashing if no text parts exist."""
    parts = response.candidates[0].content.parts if response.candidates else []
    return "".join(p.text for p in parts if hasattr(p, "text") and p.text)


def _trim_large_args(tool_args: dict, max_list: int = 20) -> dict:
    """BUG-4 FIX: trim before logging so log reflects what is actually used."""
    return {
        k: (v[:max_list] if isinstance(v, list) and len(v) > max_list else v)
        for k, v in tool_args.items()
    }


def _trim_large_result(result: dict, max_list: int = 20) -> dict:
    return {
        k: (v[:max_list] if isinstance(v, list) and len(v) > max_list else v)
        for k, v in result.items()
    }

# =========================================================
# MAIN AGENT LOOP
# =========================================================

MAX_ITERATIONS = 20
MAX_HISTORY_TURNS = 30   # BUG-5: prune history beyond this many turns


def run_workload_analysis() -> str:
    # BUG-2 FIX: ensure indexes exist before any DB queries
    ensure_indexes()

    print("\n" + "="*60)
    print("  GEMINI WORKLOAD BALANCING ENGINE STARTED")
    print("="*60 + "\n")

    user_prompt = """
    Analyse the current engineering team workload.

    Steps (follow in order):
    1. Call get_team_workload — identify overloaded employees and burnout risks.
       Note the exact values: utilization_pct, effective_hours, burnout_score,
       total_overtime_7d, avg_hours_per_day, burnout_signal_days for each person.
    2. Call get_overtime_data — cross-check burnout signals.
    3. Call get_urgent_deadlines — note tasks/projects due soon.
    4. For each overloaded employee, call get_reassignable_tasks.
    5. For each reassignable task, call find_best_candidates with that task's
       required_skills and the current assignee's employee_id.
       Note the candidate's skill_score, composite_score, utilization_pct,
       burnout_risk for inclusion in the summary.
    6. Call save_redistribution_suggestion for every strong candidate found.
    7. Return a final report following the FINAL SUMMARY FORMAT in your instructions.
       IMPORTANT: include the actual numeric values (utilization %, burnout score,
       overtime hours, avg hrs/day, skill score, composite score) for every
       employee and every task redistribution listed. Do not omit numbers.
    """

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=_convert_tools_for_gemini(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    history: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=user_prompt)])
    ]

    iteration = 0

    while True:
        iteration += 1

        if iteration > MAX_ITERATIONS:
            log.warning("Max iterations (%d) reached — stopping.", MAX_ITERATIONS)
            print("Max iterations reached — stopping.")
            break

        # BUG-5 FIX: prune history to avoid context-window overflow
        if len(history) > MAX_HISTORY_TURNS * 2:
            log.warning("History too long (%d items); pruning oldest turns.", len(history))
            history = history[:1] + history[-(MAX_HISTORY_TURNS * 2 - 1):]

        print(f"\n[Iteration {iteration}] Calling Gemini…")

        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=history,
                config=config,
            )
        except Exception as exc:
            log.error("Gemini API call failed: %s", exc)
            print(f"Gemini API call failed: {exc}")
            break

        if not response.candidates:
            log.error("Gemini returned no candidates.")
            break

        history.append(response.candidates[0].content)

        # =====================================================
        # PARSE FUNCTION CALLS
        # =====================================================

        function_calls = []
        try:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    function_calls.append(part.function_call)
        except Exception as exc:
            log.error("Error parsing response parts: %s", exc)
            break

        # =====================================================
        # NO FUNCTION CALLS → FINAL ANSWER
        # =====================================================

        if not function_calls:
            final_text = _safe_text(response)
            print("\n" + "="*60)
            print("  FINAL GEMINI ANALYSIS")
            print("="*60)
            print(final_text)
            print("="*60 + "\n")
            return final_text

        # =====================================================
        # EXECUTE TOOL CALLS
        # =====================================================

        tool_response_parts = []

        for fc in function_calls:
            tool_name = fc.name
            # BUG-4 FIX: trim before logging
            tool_args = _trim_large_args(dict(fc.args) if fc.args else {})

            log.info("Tool call → %s | args: %s", tool_name, json.dumps(tool_args, default=str))
            print(f"\n→ Tool: {tool_name}")
            print(f"  Args: {json.dumps(tool_args, indent=2, default=str)}")

            t0         = time.perf_counter()
            raw_result = run_tool(tool_name, tool_args)
            elapsed    = round(time.perf_counter() - t0, 3)

            result = _trim_large_result(raw_result)
            log.info("Tool %s completed in %.3fs", tool_name, elapsed)
            print(f"  Result preview ({elapsed}s): {str(result)[:300]}")

            tool_response_parts.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        name=tool_name,
                        response={"content": result},
                    )
                )
            )

        history.append(
            types.Content(role="tool", parts=tool_response_parts)
        )

    return "Analysis ended before a final summary was produced."


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s"
    )
    run_workload_analysis()
