import google.genai as genai
import os
import json
from schemas import JDData
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

def parse_jd(jd_text: str) -> dict:
    """
    Uses Gemini to extract structured hiring requirements from a job description.
    Returns a validated dict matching the JDData schema.
    """

    prompt = f"""
    You are an expert job description parser.

    Extract structured hiring requirements from the job description.

    IMPORTANT RULES:
    - Return ONLY valid JSON — no markdown, no explanation
    - Never return null values.
    - If a field is missing, return:
        - "" for strings
        - [] for lists
        - 0 for numbers
    - Skills must be lowercase
    - Remove duplicates
    - Experience must be numeric minimum years (e.g., 3)

    SCHEMA:
    {{
    "required_skills": ["string"],
    "preferred_skills": ["string"],
    "required_experience": number,
    "job_role": "string",
    "education_required": "string",
    "responsibilities": ["string"]
    }}

    FIELD DEFINITIONS:
    - required_skills: must-have skills explicitly listed
    - preferred_skills: good-to-have / nice-to-have skills
    - required_experience: minimum years of experience required
    - job_role: job title (lowercase, e.g., "backend developer")
    - education_required: required degree (short string)
    - responsibilities: key job responsibilities (concise)

    EXTRACTION RULES:
    - Separate required vs preferred skills carefully
    - If experience is a range (e.g., 3–6 years), use the minimum
    - Keep responsibilities short and meaningful

    Job Description:
    {jd_text}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt,
            config={"temperature": 0, "top_p": 0.1},
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)
        validated = JDData(**data)
        return validated.model_dump()

    except json.JSONDecodeError as e:
        print(f"[jd_parser] JSON decode error: {e}")
    except Exception as e:
        print(f"[jd_parser] Error: {e}")

    return JDData(
        required_skills=[], preferred_skills=[], required_experience=0.0,
        job_role="", education_required="", responsibilities=[]
    ).model_dump()
