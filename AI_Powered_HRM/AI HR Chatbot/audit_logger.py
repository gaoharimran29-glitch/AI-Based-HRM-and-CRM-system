"""
audit_logger.py
===============
Writes a structured JSON audit log for every chat interaction.

Each line in hr_audit.log is a self-contained JSON record:
  {
    "timestamp": "2026-04-28T18:30:00.123456",
    "employee_id": "emp_001",
    "event": "query" | "guardrail_blocked" | "error",
    "query": "...",          # always present
    "response": "...",       # present on successful query
    "violation": "...",      # present when guardrail blocked
    "error": "..."           # present on unexpected error
  }

Why JSON-lines? Easy to grep, pipe into jq, or load into pandas/Elasticsearch.
"""

import json
import logging
import os
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# SETUP FILE LOGGER
# ─────────────────────────────────────────────
LOG_FILE = os.getenv("AUDIT_LOG_FILE", "hr_audit.log")

_audit_logger = logging.getLogger("hr_audit")
_audit_logger.setLevel(logging.INFO)
_audit_logger.propagate = False  # don't bleed into root logger

if not _audit_logger.handlers:
    _fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    _fh.setFormatter(logging.Formatter("%(message)s"))   # raw JSON, no prefix
    _audit_logger.addHandler(_fh)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(record: dict):
    _audit_logger.info(json.dumps(record, ensure_ascii=False))


# ─────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────
def log_query(employee_id: str, query: str, response: str):
    """Log a successful LLM interaction."""
    _write({
        "timestamp":   _now(),
        "employee_id": employee_id,
        "event":       "query",
        "query":       query,
        "response":    response,
    })


def log_blocked(employee_id: str, query: str, violation: str):
    """Log a message that was blocked by a guardrail."""
    _write({
        "timestamp":   _now(),
        "employee_id": employee_id,
        "event":       "guardrail_blocked",
        "query":       query,
        "violation":   violation,
    })


def log_error(employee_id: str, query: str, error: str):
    """Log an unexpected runtime error."""
    _write({
        "timestamp":   _now(),
        "employee_id": employee_id,
        "event":       "error",
        "query":       query,
        "error":       error,
    })