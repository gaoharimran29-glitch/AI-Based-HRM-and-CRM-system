from typing import List, Dict
import numpy as np

from embeddings import get_embedding, get_embeddings_batch, cosine_similarity

# -----------------------------
# UTILITIES
# -----------------------------

def batch_embeddings(texts: List[str]) -> List[np.ndarray]:
    """Vectorized batch embed — replaces the old one-at-a-time loop."""
    filtered = [t for t in texts if t]
    return get_embeddings_batch(filtered)


def average_best_similarity(source_vecs, target_vecs):
    if not source_vecs or not target_vecs:
        return 0.0

    scores = []
    for t_vec in target_vecs:
        sims = [cosine_similarity(s_vec, t_vec) for s_vec in source_vecs]
        scores.append(max(sims))

    return float(np.mean(scores))


# -----------------------------
# JD CACHE (IMPORTANT)
# -----------------------------

jd_cache = {}


def preprocess_jd(jd: dict):
    """
    Build all JD embeddings in as few model calls as possible.
    We collect every string that needs embedding, run ONE batched
    encode(), then assign the resulting vectors back to their fields.
    """
    required_skills  = jd.get("required_skills", [])
    preferred_skills = jd.get("preferred_skills", [])
    responsibilities = jd.get("responsibilities", [])
    job_role         = jd.get("job_role", "")
    education_req    = jd.get("education_required", "")

    # ── single batched encode for all JD text ──────────────────────────
    all_texts = required_skills + preferred_skills + responsibilities + [job_role, education_req]
    all_vecs  = get_embeddings_batch([t for t in all_texts if t])

    # slice the flat vector list back into its groups
    idx = 0

    def take(n):
        nonlocal idx
        chunk = all_vecs[idx:idx + n]
        idx += n
        return chunk

    req_vecs  = take(len(required_skills))
    pref_vecs = take(len(preferred_skills))
    resp_vecs = take(len(responsibilities))
    role_vec  = all_vecs[idx]     if job_role      else np.zeros(384)
    edu_vec   = all_vecs[idx + 1] if education_req else np.zeros(384)

    return {
        "required_skills":      required_skills,
        "required_skills_vec":  req_vecs,

        "preferred_skills":     preferred_skills,
        "preferred_skills_vec": pref_vecs,

        "role_vec":             role_vec,

        "responsibilities":     responsibilities,
        "resp_vecs":            resp_vecs,

        "education_required":   education_req,
        "edu_vec":              edu_vec,

        "required_experience":  jd.get("required_experience", 0),
    }


def get_cached_jd(jd_id: str, jd: dict):
    if jd_id not in jd_cache:
        print("Preprocessing JD (only once)...")
        jd_cache[jd_id] = preprocess_jd(jd)
    return jd_cache[jd_id]


# -----------------------------
# EDUCATION NORMALIZATION
# -----------------------------

def normalize_education(text: str) -> str:
    text = text.lower()
    if "btech" in text or "bachelor" in text or "b.sc" in text:
        return "bachelor degree"
    if "mtech" in text or "master" in text:
        return "master degree"
    return text


# -----------------------------
# SKILLS MATCHING (USES JD CACHE)
# -----------------------------

def skills_match(resume_skills, jd_skills, jd_vecs, threshold=0.75):
    res_vecs = batch_embeddings(resume_skills)

    matched = []
    missing = []

    for skill, jd_vec in zip(jd_skills, jd_vecs):
        sims = [cosine_similarity(r_vec, jd_vec) for r_vec in res_vecs]
        best = max(sims) if sims else 0

        if best >= threshold:
            matched.append(skill)
        else:
            missing.append(skill)

    score = average_best_similarity(res_vecs, jd_vecs)
    return score, matched, missing


# -----------------------------
# ROLE SIMILARITY (USES JD CACHE)
# -----------------------------

def role_similarity(resume_roles: List[str], role_vec):
    if not resume_roles:
        return 0.0
    res_vecs = batch_embeddings(resume_roles)
    sims = [cosine_similarity(r_vec, role_vec) for r_vec in res_vecs]
    return max(sims) if sims else 0.0


# -----------------------------
# PROJECT ↔ RESPONSIBILITY
# -----------------------------

def project_similarity(projects: List[str], resp_vecs):
    proj_vecs = batch_embeddings(projects)
    return average_best_similarity(proj_vecs, resp_vecs)


# -----------------------------
# EDUCATION (USES NORMALIZATION)
# -----------------------------

def education_match(education: str, education_required: str):
    if not education or not education_required:
        return 1.0

    cand = normalize_education(education)
    req  = normalize_education(education_required)

    cand_vec = get_embedding(cand)
    req_vec  = get_embedding(req)

    return cosine_similarity(cand_vec, req_vec)


# -----------------------------
# EXPERIENCE
# -----------------------------

def experience_score(candidate_exp: float, required_exp: float):
    if required_exp == 0:
        return 1.0, 0

    gap   = max(0, required_exp - candidate_exp)
    score = min(candidate_exp / required_exp, 1.0)
    return score, gap


# -----------------------------
# VERDICT
# -----------------------------

def get_verdict(score: float) -> str:
    if score >= 80:
        return "Strong Match"
    elif score >= 65:
        return "Good Match"
    elif score >= 50:
        return "Moderate Match"
    elif score >= 35:
        return "Weak Match"
    else:
        return "Low Match"


# -----------------------------
# MAIN FUNCTION (USES JD CACHE)
# -----------------------------

def match_resume_to_jd(resume: Dict, jd_cached: Dict) -> Dict:

    # Pre-embed ALL resume text in one batched call
    resume_skills  = resume.get("skills", [])
    resume_roles   = resume.get("roles", [])
    resume_projects = resume.get("projects", [])

    all_resume_texts = resume_skills + resume_roles + resume_projects
    all_resume_vecs  = get_embeddings_batch([t for t in all_resume_texts if t])

    # Slice back
    skill_vecs   = all_resume_vecs[:len(resume_skills)]
    role_vecs    = all_resume_vecs[len(resume_skills):len(resume_skills) + len(resume_roles)]
    project_vecs = all_resume_vecs[len(resume_skills) + len(resume_roles):]

    # ── REQUIRED SKILLS ───────────────────────────────────────────────
    jd_req_vecs = jd_cached["required_skills_vec"]
    req_threshold = 0.7
    matched_skills, missing_skills = [], []
    req_scores = []

    for skill, jd_vec in zip(jd_cached["required_skills"], jd_req_vecs):
        sims = [cosine_similarity(r, jd_vec) for r in skill_vecs]
        best = max(sims) if sims else 0
        (matched_skills if best >= req_threshold else missing_skills).append(skill)
        req_scores.append(best)

    skill_score = float(np.mean(req_scores)) if req_scores else 0.0

    # ── PREFERRED SKILLS ──────────────────────────────────────────────
    jd_pref_vecs = jd_cached["preferred_skills_vec"]
    pref_threshold = 0.65
    pref_matched, pref_missing = [], []
    pref_scores = []

    for skill, jd_vec in zip(jd_cached["preferred_skills"], jd_pref_vecs):
        sims = [cosine_similarity(r, jd_vec) for r in skill_vecs]
        best = max(sims) if sims else 0
        (pref_matched if best >= pref_threshold else pref_missing).append(skill)
        pref_scores.append(best)

    pref_score = float(np.mean(pref_scores)) if pref_scores else 0.0

    # ── ROLE ──────────────────────────────────────────────────────────
    role_vec = jd_cached["role_vec"]
    role_sims = [cosine_similarity(r, role_vec) for r in role_vecs]
    role_score_val = max(role_sims) if role_sims else 0.0

    # ── PROJECTS ──────────────────────────────────────────────────────
    proj_score = average_best_similarity(project_vecs, jd_cached["resp_vecs"])

    # ── EDUCATION ─────────────────────────────────────────────────────
    edu_score = education_match(
        resume.get("education", ""),
        jd_cached["education_required"]
    )

    # ── EXPERIENCE ────────────────────────────────────────────────────
    candidate_exp = resume.get("experience_years", 0)
    required_exp  = jd_cached["required_experience"]

    exp_score, exp_gap = experience_score(candidate_exp, required_exp)
    exp_score = 0.7 * exp_score + 0.3 * (exp_score * role_score_val)

    if required_exp > 0 and candidate_exp > required_exp:
        bonus = min((candidate_exp - required_exp) / required_exp, 1.0)
        exp_score += 0.1 * bonus

    exp_score = min(exp_score, 1.0)

    # ── FINAL SCORE ───────────────────────────────────────────────────
    final = (
        0.30 * skill_score +
        0.15 * pref_score  +
        0.20 * proj_score  +
        0.10 * exp_score   +
        0.15 * role_score_val +
        0.10 * edu_score
    )

    # ── PENALTIES ─────────────────────────────────────────────────────
    req_total            = max(1, len(jd_cached["required_skills"]))
    req_missing_ratio    = len(missing_skills) / req_total
    req_penalty_factor   = max(0.7, 1.0 - (0.3 * req_missing_ratio))

    pref_total           = max(1, len(jd_cached["preferred_skills"]))
    pref_missing_ratio   = len(pref_missing) / pref_total
    pref_penalty_factor  = max(0.9, 1.0 - (0.1 * pref_missing_ratio))

    combined_penalty = max(0.65, req_penalty_factor * pref_penalty_factor)
    final = round(final * combined_penalty * 100)

    return {
        "candidate_name":           resume.get("candidate_name", ""),
        "email":                    resume.get("email", ""),
        "phone":                    resume.get("phone", ""),
        "score":                    final,
        "matched_skills":           matched_skills,
        "missing_skills":           missing_skills,
        "preferred_matched_skills": pref_matched,
        "preferred_missing_skills": pref_missing,
        "skill_score":              round(skill_score, 3),
        "preferred_skill_score":    round(pref_score, 3),
        "role_score":               round(role_score_val, 3),
        "project_score":            round(proj_score, 3),
        "experience_score":         round(exp_score, 3),
        "education_score":          round(edu_score, 3),
        "experience_gap":           exp_gap,
        "verdict":                  get_verdict(final),
    }


# -----------------------------
# TEST RUN
# -----------------------------

if __name__ == "__main__":
    jd_id = "backend_1"

    jd = {
        'required_skills': ['python', 'django', 'fastapi', 'restful apis', 'microservices architecture', 'postgresql', 'mysql', 'git', 'software design patterns'],
        'preferred_skills': ['docker', 'aws', 'gcp', 'azure'],
        'required_experience': 3.0,
        'job_role': 'senior backend developer',
        'education_required': "bachelor's degree",
        'responsibilities': ['build backend systems', 'develop APIs']
    }

    jd_cached = get_cached_jd(jd_id, jd)

    resume = {
        'skills': ['python', 'docker', 'aws', 'git'],
        'experience_years': 5.0,
        'education': 'BTech Computer Science',
        'projects': ['Built REST APIs using Flask'],
        'roles': ['backend developer']
    }

    result = match_resume_to_jd(resume, jd_cached)
    print(result)
