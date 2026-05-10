import logging
import os
import signal
import sys
import time

import schedule
from pymongo import MongoClient

from ai_engine import run_workload_analysis

# =========================================================
# CONFIG
# =========================================================

INTERVAL_HOURS = int(os.getenv("SCHEDULE_INTERVAL_HOURS", "6"))
MONGO_URI      = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME        = os.getenv("DB_NAME",    "workload_balancing_ai")
LOG_LEVEL      = os.getenv("LOG_LEVEL",  "INFO").upper()

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s  %(levelname)s  %(message)s",
    handlers=[
        logging.FileHandler("workload_engine.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# =========================================================
# DB LOG  (BUG-2 FIX — lazy init)
# =========================================================

_db = None

def _run_logs():
    global _db
    if _db is None:
        _db = MongoClient(MONGO_URI)[DB_NAME]
    return _db["engine_run_logs"]


def log_run(status: str, summary: str, error: str = None):
    from datetime import datetime
    try:
        _run_logs().insert_one({
            "run_at":  datetime.utcnow(),
            "status":  status,
            "summary": summary,
            "error":   error,
        })
    except Exception as exc:
        log.warning("Could not write run log to MongoDB: %s", exc)

# =========================================================
# JOB
# =========================================================

def run_engine():
    log.info("Workload Engine job triggered.")
    try:
        summary = run_workload_analysis()
        log.info("Engine completed successfully.")
        log_run("success", summary)
    except Exception as exc:
        log.exception("Engine failed: %s", exc)
        log_run("failed", "", str(exc))

# =========================================================
# GRACEFUL SHUTDOWN  (GAP-1 FIX)
# =========================================================

_shutdown = False

def _handle_signal(signum, frame):
    global _shutdown
    log.info("Received signal %s — shutting down gracefully.", signum)
    _shutdown = True

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# =========================================================
# SCHEDULE
# =========================================================

schedule.every(INTERVAL_HOURS).hours.do(run_engine)

# =========================================================
# START
# =========================================================

if __name__ == "__main__":
    log.info("Scheduler started. Engine will run every %d hour(s).", INTERVAL_HOURS)
    run_engine()   # run once immediately on start

    while not _shutdown:
        schedule.run_pending()
        time.sleep(60)

    log.info("Scheduler stopped.")
