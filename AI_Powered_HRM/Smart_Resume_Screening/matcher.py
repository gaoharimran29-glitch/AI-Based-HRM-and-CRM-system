from typing import List, Dict
import numpy as np
import re

from embeddings import (
    get_embedding,
    get_embeddings_batch,
    cosine_similarity
)

# =========================================================
# UTILITIES
# =========================================================    

def normalize_skill(skill: str) -> str:

    """
    Scalable skill normalization.
    Handles:
    - spacing variations
    - punctuation variations
    - plurals
    """

    if not skill:
        return ""

    skill = skill.strip().lower()

    # normalize spaces
    skill = re.sub(r"\s+", " ", skill)

    # remove punctuation variations
    skill = re.sub(r"[._/\\-]", " ", skill)

    # remove brackets
    skill = re.sub(r"[\(\)\[\]]", "", skill)

    # remove duplicate spaces again
    skill = re.sub(r"\s+", " ", skill)

    return skill.strip()


def enrich(text: str, role: str = "") -> str:

    base = f"Professional concept: {text}."

    if role:
        base += f" Relevant for the role of {role}."

    return base


# =========================================================
# JD CACHE
# =========================================================

jd_cache = {}


def preprocess_jd(jd: dict):

    required_skills = [
        normalize_skill(s)
        for s in jd.get("required_skills", [])
    ]

    preferred_skills = [
        normalize_skill(s)
        for s in jd.get("preferred_skills", [])
    ]

    responsibilities = jd.get("responsibilities", [])

    job_role = jd.get("job_role", "")

    education_req = jd.get("education_required", "")

    all_texts = (
        [enrich(t, job_role) for t in required_skills] +
        [enrich(t, job_role) for t in preferred_skills] +
        [enrich(t, job_role) for t in responsibilities] +
        ([enrich(job_role, job_role)] if job_role else []) +
        ([enrich(education_req, job_role)] if education_req else [])
    )

    all_vecs = get_embeddings_batch(all_texts)

    idx = 0

    def take(n):

        nonlocal idx

        chunk = all_vecs[idx:idx + n]

        idx += n

        return chunk

    req_vecs = take(len(required_skills))

    pref_vecs = take(len(preferred_skills))

    resp_vecs = take(len(responsibilities))

    role_vec = np.zeros(384)

    edu_vec = np.zeros(384)

    if job_role:
        role_vec = all_vecs[idx]
        idx += 1

    if education_req:
        edu_vec = all_vecs[idx]
        idx += 1

    return {

        "required_skills":
        required_skills,

        "required_skills_vec":
        req_vecs,

        "preferred_skills":
        preferred_skills,

        "preferred_skills_vec":
        pref_vecs,

        "responsibilities":
        responsibilities,

        "resp_vecs":
        resp_vecs,

        "role_vec":
        role_vec,

        "education_required":
        education_req,

        "edu_vec":
        edu_vec,

        "required_experience":
        jd.get("required_experience", 0),

        "job_role":
        job_role
    }


def get_cached_jd(jd_id: str, jd: dict):

    if jd_id not in jd_cache:

        print("Preprocessing JD only once...")

        jd_cache[jd_id] = preprocess_jd(jd)

    return jd_cache[jd_id]


# =========================================================
# RESUME EVIDENCE BUILDER
# =========================================================

def build_resume_evidence(resume):

    evidence = []

    # =====================================================
    # SKILLS
    # =====================================================

    for skill in resume.get("skills", []):

        skill = normalize_skill(skill)

        evidence.append({
            "type": "skill",
            "text": skill
        })

    # =====================================================
    # ROLES
    # =====================================================

    for role in resume.get("roles", []):

        evidence.append({
            "type": "role",
            "text": role
        })

    # =====================================================
    # PROJECTS
    # =====================================================

    for project in resume.get("projects", []):

        title = project.get("title", "")

        description = project.get("description", "")

        full_text = f"{title}. {description}"

        evidence.append({
            "type": "project_description",
            "text": full_text
        })

        # highlights

        for h in project.get("highlights", []):

            evidence.append({
                "type": "project_highlight",
                "text": h
            })

        # technologies

        for tech in project.get("technologies", []):

            tech = normalize_skill(tech)

            evidence.append({
                "type": "project_technology",
                "text": tech
            })

    return evidence


# =========================================================
# EMBED RESUME EVIDENCE
# =========================================================

def embed_resume_evidence(evidence, role=""):

    texts = [
        enrich(e["text"], role)
        for e in evidence
    ]

    vecs = get_embeddings_batch(texts)

    for e, v in zip(evidence, vecs):
        e["embedding"] = v

    return evidence


# =========================================================
# REQUIREMENT MATCHING
# =========================================================

def match_requirement(
    requirement,
    req_vec,
    evidence,
    threshold=0.65
):

    best_score = 0

    best_evidence = None

    for ev in evidence:

        score = cosine_similarity(
            ev["embedding"],
            req_vec
        )

        # EXACT MATCH BOOST
        if normalize_skill(ev["text"]) == normalize_skill(requirement):
            score = max(score, 0.98)

        if score > best_score:

            best_score = score

            best_evidence = ev

    return {

        "requirement":
        requirement,

        "score":
        round(float(best_score), 3),

        "matched":
        best_score >= threshold,

        "evidence":
        (
            best_evidence["text"]
            if best_evidence else ""
        ),

        "evidence_type":
        (
            best_evidence["type"]
            if best_evidence else ""
        )
    }


# =========================================================
# EDUCATION
# =========================================================

def normalize_education(text: str):

    text = text.lower()

    if (
        "btech" in text or
        "bachelor" in text or
        "b.sc" in text
    ):
        return "bachelor degree"

    if (
        "mtech" in text or
        "master" in text
    ):
        return "master degree"

    return text


def education_match(
    education: str,
    education_required: str
):

    if not education_required:
        return 1.0

    if not education:
        return 0.0

    cand = normalize_education(education)

    req = normalize_education(education_required)

    cand_vec = get_embedding(cand)

    req_vec = get_embedding(req)

    return cosine_similarity(cand_vec, req_vec)


# =========================================================
# EXPERIENCE
# =========================================================

def experience_score(
    candidate_exp,
    required_exp
):

    if required_exp == 0:
        return 1.0, 0

    gap = max(0, required_exp - candidate_exp)

    score = min(candidate_exp / required_exp, 1.0)

    return score, gap


# =========================================================
# VERDICT
# =========================================================

def get_verdict(score):

    if score >= 85:
        return "Strong Match"

    elif score >= 70:
        return "Good Match"

    elif score >= 55:
        return "Moderate Match"

    elif score >= 40:
        return "Weak Match"

    return "Low Match"


# =========================================================
# GLOBAL SIMILARITY
# =========================================================

def global_similarity(
    evidence,
    jd_cached
):

    role = jd_cached.get("job_role", "")

    resume_text = " ".join([
        e["text"]
        for e in evidence
    ])

    jd_text = " ".join(
        jd_cached["required_skills"] +
        jd_cached["preferred_skills"] +
        jd_cached["responsibilities"] +
        [role]
    )

    resume_vec = get_embedding(
        enrich(resume_text, role)
    )

    jd_vec = get_embedding(
        enrich(jd_text, role)
    )

    return cosine_similarity(
        resume_vec,
        jd_vec
    )


# =========================================================
# MAIN MATCHER
# =========================================================

def match_resume_to_jd(
    resume: Dict,
    jd_cached: Dict
):

    role = jd_cached.get("job_role", "")

    # =====================================================
    # BUILD RESUME EVIDENCE
    # =====================================================

    evidence = build_resume_evidence(resume)

    evidence = embed_resume_evidence(
        evidence,
        role
    )

    # =====================================================
    # REQUIRED SKILLS
    # =====================================================

    required_matches = []

    for skill, vec in zip(
        jd_cached["required_skills"],
        jd_cached["required_skills_vec"]
    ):

        result = match_requirement(
            skill,
            vec,
            evidence,
            threshold=0.68
        )

        required_matches.append(result)

    # =====================================================
    # PREFERRED SKILLS
    # =====================================================

    preferred_matches = []

    for skill, vec in zip(
        jd_cached["preferred_skills"],
        jd_cached["preferred_skills_vec"]
    ):

        result = match_requirement(
            skill,
            vec,
            evidence,
            threshold=0.63
        )

        preferred_matches.append(result)

    # =====================================================
    # SCORES
    # =====================================================

    required_scores = [
        r["score"]
        for r in required_matches
    ]

    preferred_scores = [
        r["score"]
        for r in preferred_matches
    ]

    skill_score = (
        float(np.mean(required_scores))
        if required_scores else 0
    )

    pref_score = (
        float(np.mean(preferred_scores))
        if preferred_scores else 0
    )

    # =====================================================
    # RESPONSIBILITY SCORE
    # =====================================================

    responsibility_scores = []

    for resp, vec in zip(
        jd_cached["responsibilities"],
        jd_cached["resp_vecs"]
    ):

        result = match_requirement(
            resp,
            vec,
            evidence,
            threshold=0.60
        )

        responsibility_scores.append(
            result["score"]
        )

    project_score = (
        float(np.mean(responsibility_scores))
        if responsibility_scores else 0
    )

    # =====================================================
    # ROLE SCORE
    # =====================================================

    role_score = 0

    if role:

        role_vec = jd_cached["role_vec"]

        role_matches = []

        for ev in evidence:

            if ev["type"] not in [
                "role",
                "project_description"
            ]:
                continue

            score = cosine_similarity(
                ev["embedding"],
                role_vec
            )

            role_matches.append(score)

        role_score = (
            max(role_matches)
            if role_matches else 0
        )

    # =====================================================
    # EDUCATION
    # =====================================================

    edu_score = education_match(
        resume.get("education", ""),
        jd_cached["education_required"]
    )

    # =====================================================
    # EXPERIENCE
    # =====================================================

    exp_score, exp_gap = experience_score(
        resume.get("experience_years", 0),
        jd_cached["required_experience"]
    )

    # =====================================================
    # GLOBAL SCORE
    # =====================================================

    global_score = global_similarity(
        evidence,
        jd_cached
    )

    # =====================================================
    # PENALTIES
    # =====================================================

    required_missing = [
        r for r in required_matches
        if not r["matched"]
    ]

    missing_ratio = (
        len(required_missing) /
        max(1, len(required_matches))
    )

    penalty = 1 - (missing_ratio * 0.25)

    penalty = max(0.75, penalty)

    # =====================================================
    # FINAL SCORE
    # =====================================================

    final = (
        0.35 * skill_score +
        0.15 * pref_score +
        0.15 * project_score +
        0.10 * role_score +
        0.10 * exp_score +
        0.05 * edu_score +
        0.10 * global_score
    )

    final = final * penalty

    final = round(final * 100)

    # =====================================================
    # MATCHED / MISSING
    # =====================================================

    matched_required = [
        r for r in required_matches
        if r["matched"]
    ]

    missing_required = [
        r for r in required_matches
        if not r["matched"]
    ]

    matched_preferred = [
        r for r in preferred_matches
        if r["matched"]
    ]

    missing_preferred = [
        r for r in preferred_matches
        if not r["matched"]
    ]

    # =====================================================
    # RETURN
    # =====================================================

    return {

        "candidate_name":
        resume.get("candidate_name", ""),

        "email":
        resume.get("email", ""),

        "phone":
        resume.get("phone", ""),

        "score":
        final,

        "verdict":
        get_verdict(final),

        "required_matches":
        matched_required,

        "required_missing":
        missing_required,

        "preferred_matches":
        matched_preferred,

        "preferred_missing":
        missing_preferred,

        "skill_score":
        round(skill_score, 3),

        "preferred_skill_score":
        round(pref_score, 3),

        "project_score":
        round(project_score, 3),

        "role_score":
        round(role_score, 3),

        "experience_score":
        round(exp_score, 3),

        "education_score":
        round(edu_score, 3),

        "global_score":
        round(global_score, 3),

        "experience_gap":
        exp_gap
    }