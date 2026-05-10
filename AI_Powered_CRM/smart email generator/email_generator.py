"""
Smart Email Generator Service
------------------------------
Generates contextual follow-up emails for CRM leads using an LLM.

HOW IT WORKS:
1. Frontend calls POST /email/generate-followup with a lead's unique_id
2. This service fetches the lead details from the ML service (same as /lead/{id})
3. Builds a prompt using lead status, past communication context, and deal stage
4. Calls the LLM API (plug in your API key when provided)
5. Returns the generated email subject + body

USAGE (add this file to your project, then register the router in main.py):
    from email_generator import router as email_router
    app.include_router(email_router)
"""

import os
import logging
import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter(prefix="/email", tags=["Email Generator"])

# ─── Request / Response Models ────────────────────────────────────────────────

class EmailGenerateRequest(BaseModel):
    unique_id: str                          # Lead's unique_id from MongoDB
    deal_stage: Optional[str] = "Prospect" # e.g. Prospect, Negotiation, Closed
    past_communication: Optional[str] = "" # Any notes / last interaction summary

class EmailGenerateResponse(BaseModel):
    success: bool
    lead_name: str
    lead_email: str
    subject: str
    body: str
    lead_temperature: str                   # Hot / Warm / Cold


# ─── LLM Configuration (plug in key when provided) ───────────────────────────

# Supported providers: "openai" | "anthropic" | "groq" | "gemini"
# Change LLM_PROVIDER and set the matching env variable when API key is given.
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY  = os.getenv("LLM_API_KEY", "")         # Set this when key is provided
LLM_MODEL    = os.getenv("LLM_MODEL", "gpt-4o-mini") # Change model as needed


# ─── Internal: Fetch lead from existing ML service ───────────────────────────

def _fetch_lead_from_service(unique_id: str) -> dict:
    """
    Calls the existing /lead/{unique_id} endpoint on the same FastAPI server
    to get full lead details with ML predictions.
    """
    try:
        # Import the ML service directly to avoid HTTP overhead (same process)
        from main import get_ml_service
        ml_service = get_ml_service()
        lead = ml_service.get_lead_with_prediction(unique_id)
        if not lead:
            raise HTTPException(status_code=404, detail=f"Lead '{unique_id}' not found")
        return lead
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[EmailGen] Failed to fetch lead {unique_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch lead details")


# ─── Internal: Build LLM Prompt ──────────────────────────────────────────────

def _build_prompt(lead: dict, deal_stage: str, past_communication: str) -> str:
    """
    Constructs the prompt sent to the LLM.
    Uses: Lead status (temperature), past communication, deal stage.
    """
    name       = lead.get("name", "the lead")
    role       = lead.get("role_position", "Professional")
    skills     = lead.get("skills", "N/A")
    location   = lead.get("location", "N/A")
    experience = lead.get("years_of_experience", 0)
    temperature = lead.get("ml_prediction", {}).get("predicted_temperature", "Warm")

    # Tone guidance based on lead temperature
    tone_map = {
        "Hot":  "urgent and enthusiastic — this lead is highly interested, push toward action",
        "Warm": "friendly and informative — this lead is considering, nurture their interest",
        "Cold": "polite and value-focused — this lead needs re-engagement, remind them of benefits",
    }
    tone = tone_map.get(temperature, "professional and helpful")

    prompt = f"""You are a professional B2B sales executive writing a follow-up email for a CRM lead.

LEAD DETAILS:
- Name: {name}
- Role/Position: {role}
- Skills: {skills}
- Location: {location}
- Years of Experience: {experience}
- Lead Temperature (AI Score): {temperature}
- Current Deal Stage: {deal_stage}

PAST COMMUNICATION SUMMARY:
{past_communication if past_communication else "No previous communication recorded."}

INSTRUCTIONS:
Write a short, personalized follow-up email. Tone should be: {tone}.
The email must:
1. Address the lead by their first name
2. Reference their role/deal stage naturally
3. Include a clear call-to-action (schedule a call, reply, etc.)
4. Be 3-5 short paragraphs max

Return ONLY in this exact format (no extra text):
SUBJECT: <subject line here>
BODY:
<email body here>
"""
    return prompt


# ─── Internal: Call the LLM ──────────────────────────────────────────────────

async def _call_llm(prompt: str) -> str:
    """
    Calls the configured LLM provider.
    Currently supports OpenAI-compatible APIs.
    Add more providers below when needed.
    """
    if not LLM_API_KEY:
        # Return a placeholder so the feature works without a key (for testing)
        logging.warning("[EmailGen] LLM_API_KEY not set — returning placeholder email")
        return (
            "SUBJECT: Following up on our conversation\n"
            "BODY:\nHi [Lead Name],\n\n"
            "This is a placeholder email. Please set LLM_API_KEY in your environment "
            "variables to enable AI-generated emails.\n\n"
            "Best regards,\nYour CRM"
        )

    if LLM_PROVIDER == "openai":
        return await _call_openai(prompt)
    elif LLM_PROVIDER == "groq":
        return await _call_groq(prompt)
    elif LLM_PROVIDER == "anthropic":
        return await _call_anthropic(prompt)
    else:
        raise HTTPException(status_code=500, detail=f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


async def _call_openai(prompt: str) -> str:
    """OpenAI / OpenAI-compatible API (also works for Groq with base_url change)."""
    base_url = "https://api.openai.com/v1/chat/completions"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            base_url,
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 600,
            }
        )
    if response.status_code != 200:
        logging.error(f"[EmailGen] OpenAI error: {response.text}")
        raise HTTPException(status_code=502, detail="LLM API call failed")

    return response.json()["choices"][0]["message"]["content"].strip()


async def _call_groq(prompt: str) -> str:
    """Groq API — just change base_url and model."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {LLM_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL or "llama3-8b-8192",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 600,
            }
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Groq API call failed")
    return response.json()["choices"][0]["message"]["content"].strip()


async def _call_anthropic(prompt: str) -> str:
    """Anthropic Claude API."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": LLM_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": LLM_MODEL or "claude-3-haiku-20240307",
                "max_tokens": 600,
                "messages": [{"role": "user", "content": prompt}],
            }
        )
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Anthropic API call failed")
    return response.json()["content"][0]["text"].strip()


# ─── Internal: Parse LLM Output ──────────────────────────────────────────────

def _parse_email_output(raw: str) -> tuple[str, str]:
    """
    Parses LLM output into (subject, body).
    Expected format:
        SUBJECT: <subject>
        BODY:
        <body text>
    """
    subject = "Follow-up"
    body = raw

    lines = raw.strip().splitlines()
    subject_line = next((l for l in lines if l.upper().startswith("SUBJECT:")), None)
    if subject_line:
        subject = subject_line.split(":", 1)[1].strip()

    body_start = next((i for i, l in enumerate(lines) if l.upper().startswith("BODY:")), None)
    if body_start is not None:
        body = "\n".join(lines[body_start + 1:]).strip()

    return subject, body


# ─── Main Endpoint ────────────────────────────────────────────────────────────

@router.post("/generate-followup", response_model=EmailGenerateResponse,
             summary="Generate Follow-up Email for a Lead")
async def generate_followup_email(request: EmailGenerateRequest):
    """
    Generates a contextual follow-up email for the given lead.

    Steps:
    1. Fetch lead details from the ML service using unique_id
    2. Build a prompt using lead status (temperature), deal stage, past communication
    3. Call the LLM to generate the email
    4. Return the subject and body

    Set LLM_API_KEY environment variable to enable real generation.
    When no key is set, returns a placeholder email (useful for testing).
    """
    # Step 1: Get lead
    lead = _fetch_lead_from_service(request.unique_id)

    # Step 2: Build prompt
    prompt = _build_prompt(lead, request.deal_stage, request.past_communication)

    # Step 3: Call LLM
    raw_output = await _call_llm(prompt)

    # Step 4: Parse output
    subject, body = _parse_email_output(raw_output)

    temperature = lead.get("ml_prediction", {}).get("predicted_temperature", "Unknown")

    return EmailGenerateResponse(
        success=True,
        lead_name=lead.get("name", ""),
        lead_email=lead.get("email", ""),
        subject=subject,
        body=body,
        lead_temperature=temperature,
    )