import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from text_extractor import extract_text_from_pdf
from jd_parser import parse_jd
from resume_parser import parse_resume
from matcher import match_resume_to_jd, get_cached_jd

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG  — edit these for your use-case
# ──────────────────────────────────────────────────────────────────────────────

JD_PATH = r"C:\Users\hp\Desktop\HRM\AI_Powered_HRM\Smart_Resume_Screening\resumes\Job Title.pdf"

# Folder containing candidate PDFs  (or pass a list of paths to main())
RESUME_FOLDER = r"AI_Powered_CRM\backend\services\Smart_Resume_Screening\resumesjjd"

# Fallback single-resume path (used when RESUME_FOLDER doesn't exist / is empty)
SINGLE_RESUME_PATH = r"C:\Users\hp\Desktop\HRM\AI_Powered_HRM\Smart_Resume_Screening\resumes\Gaohar Imran Resume.pdf"

# How many resumes to process in parallel.
# Gemini free-tier allows ~15 req/min → keep this ≤ 15 to avoid 429s.
# Paid tier: bump to 20-30.
MAX_WORKERS = 10


# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def generate_jd_id(jd_text: str) -> str:
    return hashlib.md5(jd_text.encode()).hexdigest()


def collect_resume_paths(folder: str, single_fallback: str) -> List[str]:
    """Return a list of PDF paths to process."""
    folder_path = Path(folder)
    if folder_path.is_dir():
        paths = sorted(str(p) for p in folder_path.glob("*.pdf"))
        if paths:
            return paths
    # fallback to single resume
    return [single_fallback]


# ──────────────────────────────────────────────────────────────────────────────
# JD  (run once, before spawning threads)
# ──────────────────────────────────────────────────────────────────────────────

def process_jd(jd_path: str) -> dict:
    print("Extracting JD text...")
    jd_text = extract_text_from_pdf(jd_path)

    jd_id = generate_jd_id(jd_text)

    print("Parsing JD with Gemini...")
    jd = parse_jd(jd_text)

    print("⚡ Embedding JD fields (once)...")
    jd_cached = get_cached_jd(jd_id, jd)

    print("JD ready.\n")
    return jd_cached


# ──────────────────────────────────────────────────────────────────────────────
# PER-RESUME WORKER  (called in parallel)
# ──────────────────────────────────────────────────────────────────────────────

def process_single_resume(resume_path: str, jd_cached: dict) -> dict:
    """
    Full pipeline for ONE resume.  Designed to be called from a thread pool.
    Returns a result dict (never raises — errors are captured as a field).
    """
    resume_name = Path(resume_path).name
    try:
        # 1. PDF → raw text
        resume_text = extract_text_from_pdf(resume_path)

        # 2. Gemini parse (I/O-bound network call)
        resume = parse_resume(resume_text)

        # 3. Embedding + matching (CPU-bound but releases GIL)
        result = match_resume_to_jd(resume, jd_cached)
        result["resume"] = resume_name
        result["status"] = "ok"
        return result

    except Exception as exc:
        return {
            "resume": resume_name,
            "status": "error",
            "error": str(exc),
            "score": -1,
            "verdict": "Error",
        }


# ──────────────────────────────────────────────────────────────────────────────
# PARALLEL BATCH  (the main speed gain)
# ──────────────────────────────────────────────────────────────────────────────

def process_resumes_parallel(
    resume_paths: List[str],
    jd_cached: dict,
    max_workers: int = MAX_WORKERS,
) -> List[dict]:
    """
    Process all resumes in parallel using a thread pool.
    Results are returned sorted by score (descending).
    """
    results = []
    total = len(resume_paths)

    print(f"Processing {total} resume(s) with up to {max_workers} parallel workers...\n")
    t0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_single_resume, path, jd_cached): path
            for path in resume_paths
        }

        # Collect as each one finishes (not necessarily in order)
        for i, future in enumerate(as_completed(future_to_path), start=1):
            result = future.result()
            results.append(result)

            status_icon = "✅" if result["status"] == "ok" else "❌"
            print(
                f"  {status_icon} [{i}/{total}] {result['resume']} "
                f"— score: {result.get('score', 'N/A')} "
                f"({result.get('verdict', '')})"
            )

    elapsed = time.perf_counter() - t0
    print(f"\n⏱  Finished {total} resume(s) in {elapsed:.1f}s "
          f"({elapsed/total:.2f}s per resume)\n")

    # Sort by score descending; errors go to the bottom
    results.sort(key=lambda r: r.get("score", -1), reverse=True)
    return results


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def save_results(results: List[dict], filename: Optional[str] = None) -> str:
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"match_results_{timestamp}.json"

    base_dir = Path(__file__).resolve().parent

    output_dir = base_dir / "outputs"
    output_dir.mkdir(exist_ok=True)

    file_path = output_dir / filename

    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {file_path}")
    return str(file_path)


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────

def main(
    jd_path: str = JD_PATH,
    resume_paths: Optional[List[str]] = None,
    max_workers: int = MAX_WORKERS,
):
    print("=" * 60)
    print("  Smart Resume Screening — Parallel Pipeline")
    print("=" * 60 + "\n")

    # ── Step 1: JD (once) ────────────────────────────────────────────
    jd_cached = process_jd(jd_path)

    # ── Step 2: collect resume paths ─────────────────────────────────
    if resume_paths is None:
        resume_paths = collect_resume_paths(RESUME_FOLDER, SINGLE_RESUME_PATH)

    if not resume_paths:
        print("No resume PDFs found. Check RESUME_FOLDER or SINGLE_RESUME_PATH.")
        return []

    # ── Step 3: parallel processing ──────────────────────────────────
    results = process_resumes_parallel(resume_paths, jd_cached, max_workers)

    # ── Step 4: print leaderboard ────────────────────────────────────
    print("RANKED RESULTS:")
    print("-" * 60)
    for rank, r in enumerate(results, start=1):
        if r["status"] == "ok":
            print(f"  #{rank:>3}  {r['resume']:<35} {r['score']:>3}/100  {r['verdict']}")
        else:
            print(f"  #{rank:>3}  {r['resume']:<35} ERROR: {r.get('error','')}")
    print("-" * 60)

    # ── Step 5: save ─────────────────────────────────────────────────
    save_results(results)

    return results


if __name__ == "__main__":
    main()