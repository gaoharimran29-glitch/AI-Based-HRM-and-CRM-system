import json
import io
import joblib
import pandas as pd
import pdfplumber
import docx
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
from contextlib import asynccontextmanager
from feature_engineer import LeadFeatureEngineer
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

ml_model = None
llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model, llm
    try:
        ml_model = joblib.load("catboost_lead_scoring_model2.pkl")
        print("✅ ML model loaded.")
    except FileNotFoundError:
        raise RuntimeError("catboost_lead_scoring_model.pkl not found. Place it in the same directory.")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
    print("✅ Gemini LLM ready.")
    yield

app = FastAPI(
    title="Lead Scoring & AI Insights API",
    description="Scores leads with an ML model and generates AI-powered insights from attached documents.",
    version="1.0.0",
    lifespan=lifespan,
)

class LeadScoringResponse(BaseModel):
    lead_id: str
    conversion_probability: float = Field(..., description="ML-predicted probability (0.0 – 1.0)")
    lead_tier: str = Field(..., description="Hot / Warm / Cold classification")
    ai_summary: str = Field(..., description="AI-generated executive summary from the uploaded document")
    recommended_actions: List[str] = Field(..., description="3 tactical next-step recommendations")


def assign_lead_tier(prob: float) -> str:
    if prob >= 0.70:
        return "🔥 Hot"
    elif prob >= 0.30:
        return "⚡ Warm "
    return "❄️ Cold"

def extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract plain text from PDF or DOCX bytes."""
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        text_parts = []
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        if not text_parts:
            raise HTTPException(status_code=422, detail="PDF appears to be scanned/image-only; no text could be extracted.")
        return "\n".join(text_parts)

    elif ext == "docx":
        doc = docx.Document(io.BytesIO(content))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        if not paragraphs:
            raise HTTPException(status_code=422, detail="DOCX appears to be empty.")
        return "\n".join(paragraphs)

    else:
        raise HTTPException(status_code=415, detail=f"Unsupported file type '.{ext}'. Please upload a PDF or DOCX.")


def run_ml_scoring(
    budget: float,
    meet_count: int,
    website_visits: int,
    mail_open_rate: float,
    mail_response_count: int,
    industry: str,
) -> float:
    """Feed sanitised features into the CatBoost model."""
    # Clamp adversarial / corrupted values
    budget = max(0.0, budget)
    meet_count = max(0, meet_count)
    website_visits = max(0, website_visits)
    mail_open_rate = max(0.0, min(1.0, mail_open_rate))
    mail_response_count = max(0, mail_response_count)

    features = pd.DataFrame([{
        "industry": industry,
        "budget": budget,
        "website_visits": website_visits,
        "mail_response_count": mail_response_count,
        "total_meet_count": meet_count,
        "mail_open_rate": mail_open_rate,
    }])

    prob = float(ml_model.predict_proba(features)[:, 1][0])
    return round(prob, 4)


def generate_ai_insights(lead_metrics: dict, doc_text: str,
) -> dict:
    prompt = f"""
You are an elite B2B Enterprise Sales Strategist Agent. Analyze the following lead profile and
attached document text to generate an executive summary and tactical next-step recommendations.

--- LEAD STRUCTURAL METRICS ---
- Lead ID: {lead_metrics['lead_id']}
- Customer Budget: ${lead_metrics['budget']:,}
- AI Predicted Conversion Probability: {lead_metrics['conversion_probability']:.2%}
- Heuristic Lead Quality Score: {lead_metrics['lead_quality_score']}/100
- Pipeline Activity: {lead_metrics['meet_count']} meetings, {lead_metrics['web_visits']} website visits, {lead_metrics['mail_responses']} mail responses.

--- EXTRACTED ATTACHED DOCUMENTATION ---
{doc_text[:6000]}

--- OUTPUT FORMAT ---
Your response must be a valid JSON object strictly matching this schema. Do NOT include markdown
formatting (no ```json fences, no extra text outside the JSON).
{{
    "ai_summary": "A concise 7-8 sentence summary highlighting who the lead is, their primary business pain points extracted from the document, and their current buying signals.",
    "recommended_actions": [
        "Action 1: Immediate specific next step based on the document text.",
        "Action 2: Tactical advice based on their metrics.",
        "Action 3: Risk mitigation step."
    ]
}}
"""
    response = llm.invoke(prompt)
    raw = response.content.strip()
    # Strip accidental markdown fences just in case
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


@app.post("/score-lead", response_model=LeadScoringResponse, summary="Score a lead and get AI insights")
async def score_lead(
    lead_id: str = Form(..., description="Unique lead identifier"),
    industry: str = Form(..., description="Lead's industry (e.g. Finance, SaaS, Healthcare)"),
    budget: float = Form(..., description="Customer budget in USD"),
    meet_count: int = Form(..., description="Total number of meetings held"),
    website_visits: int = Form(..., description="Total website visits"),
    mail_open_rate: float = Form(..., description="Email open rate (0.0 – 1.0)"),
    mail_response_count: int = Form(..., description="Number of mail responses received"),
    document: UploadFile = File(..., description="Supporting document — PDF or DOCX"),
):
    file_bytes = await document.read()
    doc_text = extract_text_from_file(document.filename, file_bytes)

    conversion_prob = run_ml_scoring(
        budget=budget,
        meet_count=meet_count,
        website_visits=website_visits,
        mail_open_rate=mail_open_rate,
        mail_response_count=mail_response_count,
        industry=industry,
    )
    lead_tier = assign_lead_tier(conversion_prob)

    lead_quality_score = round(conversion_prob * 100)

    lead_metrics = {
        "lead_id": lead_id,
        "budget": budget,
        "conversion_probability": conversion_prob,
        "lead_quality_score": lead_quality_score,
        "meet_count": meet_count,
        "web_visits": website_visits,
        "mail_responses": mail_response_count,
    }

    insights = generate_ai_insights(lead_metrics, doc_text)

    return LeadScoringResponse(
        lead_id=lead_id,
        conversion_probability=conversion_prob,
        lead_tier=lead_tier,
        ai_summary=insights["ai_summary"],
        recommended_actions=insights["recommended_actions"],
    )


@app.get("/health", summary="Health check")
def health():
    return {"status": "ok", "model_loaded": ml_model is not None}