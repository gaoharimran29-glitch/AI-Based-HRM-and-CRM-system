import google.genai as genai
import os
import json
import re

from schemas import ResumeData
from dotenv import load_dotenv

load_dotenv()

# =========================================================
# GEMINI CLIENT
# =========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# =========================================================
# HELPERS
# =========================================================

from json import JSONDecoder


def extract_first_json(text: str):
    """
    Robustly extracts the FIRST valid JSON object
    from Gemini output.
    """

    # Remove markdown fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    decoder = JSONDecoder()

    start = text.find("{")

    if start == -1:
        raise ValueError("No JSON object found")

    obj, end = decoder.raw_decode(text[start:])

    return obj

def clean_json_response(text: str) -> str:
    """
    Removes markdown code fences if Gemini returns them.
    """

    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)

    return text.strip()


def deduplicate_skills(skills):
    """
    Lowercase + deduplicate skills.
    """

    seen = set()
    cleaned = []

    for skill in skills:
        if not isinstance(skill, str):
            continue

        s = skill.strip().lower()

        if s and s not in seen:
            seen.add(s)
            cleaned.append(s)

    return cleaned


def clean_projects(projects):
    """
    Cleans project objects.
    """

    cleaned_projects = []

    for project in projects:

        if not isinstance(project, dict):
            continue

        title = str(project.get("title", "")).strip()

        description = str(
            project.get("description", "")
        ).strip()

        highlights = project.get("highlights", [])
        technologies = project.get("technologies", [])

        if not isinstance(highlights, list):
            highlights = []

        if not isinstance(technologies, list):
            technologies = []

        highlights = [
            str(h).strip()
            for h in highlights
            if str(h).strip()
        ]

        technologies = deduplicate_skills(technologies)

        cleaned_projects.append({
            "title": title,
            "description": description,
            "highlights": highlights,
            "technologies": technologies
        })

    return cleaned_projects

# =========================================================
# MAIN PARSER
# =========================================================

def parse_resume(resume_text: str) -> dict:
    """
    Parses resume using Gemini and returns structured JSON.
    """

    prompt = f"""
You are a highly accurate ATS resume parsing system.

Your task is to extract structured information from the given resume.

IMPORTANT RULES:
- Return ONLY valid JSON
- No markdown
- No explanations
- No extra text
- If data is missing return:
  - [] for lists
  - "" for strings
  - 0 for numbers
- Do NOT hallucinate
- Do NOT invent information
- Skills and technologies MUST be lowercase
- Remove duplicate skills
- Extract experience as TOTAL years only
- Contact details must only be extracted if clearly visible

=========================================================
OUTPUT JSON SCHEMA
=========================================================

{{
  "candidate_name": "string",

  "email": "string",

  "phone": "string",

  "skills": ["string"],

  "experience_years": number,

  "companies": ["string"],

  "education": "string",

  "projects": [
    {{
      "title": "string",

      "description": "string",

      "highlights": [
        "string"
      ],

      "technologies": [
        "string"
      ]
    }}
  ],

  "roles": ["string"]
}}

=========================================================
FIELD DEFINITIONS
=========================================================

candidate_name:
- Full candidate name

email:
- Valid email address only

phone:
- Valid phone number only

skills:
- Technical + professional skills
- Must be lowercase
- Remove duplicates

experience_years:
- Total professional experience in years
- Numeric only

companies:
- Companies worked at

education:
- Highest qualification only

roles:
- Job roles/designations

=========================================================
PROJECT EXTRACTION RULES
=========================================================

For EACH project extract:

1. title
- Actual project title

2. description
- Concise semantic summary
- Include:
  - what was built
  - AI/ML/NLP concepts used
  - deployment details
  - evaluation details
  - outcomes/results

3. highlights
- IMPORTANT
- Extract short semantic achievement chunks
- Each highlight should be concise
- Examples:
  - "built rag pipeline"
  - "reduced hallucination"
  - "implemented llm-as-judge evaluation"
  - "deployed on aws"

4. technologies
- Lowercase only
- Include:
  - frameworks
  - databases
  - cloud services
  - ml libraries
  - ai tools
- Remove duplicates

=========================================================
EXTRACTION QUALITY RULES
=========================================================

- Do NOT copy huge paragraphs verbatim
- Preserve semantic meaning
- Keep descriptions concise but information-rich
- Ignore decorative text
- Ignore GitHub labels
- Ignore hyperlinks
- Ignore icons/symbols

=========================================================
RESUME
=========================================================

{resume_text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={
                "temperature": 0,
                "top_p": 0.1,
            },
        )

        text = clean_json_response(response.text)

        data = extract_first_json(text)

        # =================================================
        # POST PROCESSING
        # =================================================

        data["skills"] = deduplicate_skills(
            data.get("skills", [])
        )

        data["projects"] = clean_projects(
            data.get("projects", [])
        )

        # =================================================
        # VALIDATION
        # =================================================

        validated = ResumeData(**data)

        return validated.model_dump()

    except json.JSONDecodeError as e:

        print(f"[resume_parser] JSON decode error: {e}")

    except Exception as e:

        print(f"[resume_parser] Error: {e}")

    # =====================================================
    # SAFE FALLBACK
    # =====================================================

    return ResumeData(
        candidate_name="",
        email="",
        phone="",
        skills=[],
        experience_years=0.0,
        companies=[],
        education="",
        projects=[],
        roles=[]
    ).model_dump()
