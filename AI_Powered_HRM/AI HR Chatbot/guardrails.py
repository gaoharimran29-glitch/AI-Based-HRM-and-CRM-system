"""
guardrails.py
=============
All security checks happen HERE before anything touches the LLM or DB.

Layers:
  1. Input length check          — block absurdly long inputs
  2. Prompt injection detection  — catch attempts to override system prompt
  3. PII / ID leak detection     — catch "show me emp_002's data" attacks
  4. Hate / abuse filter         — block toxic messages
  5. Rate limiter                — max N requests per employee per minute
"""

import re
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_INPUT_LENGTH = 500          # characters
RATE_LIMIT_MAX   = 10           # requests
RATE_LIMIT_WINDOW = 60          # seconds

# ─────────────────────────────────────────────
# RATE LIMITER  (in-memory; good for single process)
# ─────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(employee_id: str) -> tuple[bool, str]:
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Keep only timestamps inside the current window
    _rate_store[employee_id] = [
        t for t in _rate_store[employee_id] if t > window_start
    ]

    if len(_rate_store[employee_id]) >= RATE_LIMIT_MAX:
        return False, (
            f"Too many requests. You can send at most {RATE_LIMIT_MAX} "
            f"messages per minute. Please wait a moment."
        )

    _rate_store[employee_id].append(now)
    return True, ""


# ─────────────────────────────────────────────
# PROMPT INJECTION PATTERNS
# ─────────────────────────────────────────────
_INJECTION_PATTERNS = [
    # Classic override attempts
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)",
    r"forget\s+(everything|all|your|the)\s*(previous|prior|above|instructions?)?",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+are|a|an)\s+",
    r"pretend\s+(you\s+are|to\s+be)",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[system\]",
    # Jailbreak keywords
    r"jailbreak",
    r"dan\s+mode",
    r"developer\s+mode",
    r"override\s+(safety|guardrail|restriction)",
    # Prompt leaking
    r"show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
    r"repeat\s+(your\s+)?(instructions?|prompt|rules?)",
    r"what\s+(are\s+)?your\s+(instructions?|rules?|prompt)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    flags=re.IGNORECASE
)


def _check_prompt_injection(text: str) -> tuple[bool, str]:
    if _INJECTION_RE.search(text):
        return False, (
            "Your message contains content that cannot be processed. "
            "Please ask a normal HR question."
        )
    return True, ""


# ─────────────────────────────────────────────
# CROSS-USER DATA ACCESS (ID LEAK)
# ─────────────────────────────────────────────
# Catches: "show me emp_002's salary" / "what about employee 003"
_ID_PATTERN = re.compile(
    r"\bemp[_\s]?\d{3,}\b"         # emp_001, emp003, emp 002
    r"|\bemployee\s*(id\s*)?[:#]?\s*\d+\b"  # employee id: 42
    r"|\bstaff\s*id\s*[:#]?\s*\d+\b",
    flags=re.IGNORECASE
)

def _check_id_leak(text: str, session_employee_id: str) -> tuple[bool, str]:
    """Block if the user references ANY employee ID other than their own."""
    matches = _ID_PATTERN.findall(text)
    for match in matches:
        # Normalise: strip spaces, underscores, lowercase
        normalised = re.sub(r"[\s_]", "", match).lower()
        session_norm = re.sub(r"[\s_]", "", session_employee_id).lower()
        if normalised != session_norm:
            return False, (
                "You can only access your own HR data. "
                "Requesting another employee's information is not allowed."
            )
    return True, ""


# ─────────────────────────────────────────────
# HATE / ABUSE FILTER (lightweight keyword set)
# ─────────────────────────────────────────────
_ABUSE_WORDS = {
    "fuck", "shit", "bitch", "asshole", "bastard",
    "idiot", "moron", "stupid", "kill", "die", "hate",
}

def _check_abuse(text: str) -> tuple[bool, str]:
    tokens = set(re.findall(r"\b\w+\b", text.lower()))
    if tokens & _ABUSE_WORDS:
        return False, (
            "Please keep your messages professional. "
            "Abusive language is not allowed."
        )
    return True, ""


# ─────────────────────────────────────────────
# INPUT LENGTH CHECK
# ─────────────────────────────────────────────
def _check_length(text: str) -> tuple[bool, str]:
    if len(text) > MAX_INPUT_LENGTH:
        return False, (
            f"Your message is too long ({len(text)} characters). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )
    return True, ""


# ─────────────────────────────────────────────
# PUBLIC API — single entry point
# ─────────────────────────────────────────────
class GuardrailViolation(Exception):
    """Raised when a guardrail check fails. Message is safe to show to user."""
    pass


def validate_input(text: str, employee_id: str) -> str:
    """
    Run all guardrail checks on `text` for the given `employee_id`.

    Returns the (stripped) text if all checks pass.
    Raises GuardrailViolation with a user-friendly message if any check fails.
    """
    text = text.strip()

    checks = [
        _check_length(text),
        _check_rate_limit(employee_id),
        _check_prompt_injection(text),
        _check_id_leak(text, employee_id),
        _check_abuse(text),
    ]

    for passed, message in checks:
        if not passed:
            raise GuardrailViolation(message)

    return text
