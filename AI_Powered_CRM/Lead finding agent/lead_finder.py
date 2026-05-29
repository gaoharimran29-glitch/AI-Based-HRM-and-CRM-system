import os
import json
import logging
from typing import Optional
import re
import sys

import requests
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# ──────────────────────────────────────────────────────────────
# App & Logging
# ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lead Generation API",
    description="Searches Google Maps via SerpAPI, scrapes contact emails, and scores business leads.",
    version="2.0.0",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────

SERPAPI_KEY: Optional[str] = os.getenv("SERP_API_KEY")
SERPAPI_URL = "https://serpapi.com/search.json"
REQUEST_TIMEOUT = 15

SCORE_HOT  = 80
SCORE_WARM = 50

WEIGHT_WEBSITE_PRESENT = 30
WEIGHT_WEBSITE_ABSENT  = 10
WEIGHT_PHONE           = 20
WEIGHT_EMAIL           = 20
WEIGHT_RATING_HIGH     = 20   # rating >= 4.5
WEIGHT_RATING_MID      = 10   # rating >= 4.0
WEIGHT_REVIEWS_HIGH    = 15   # reviews > 100
WEIGHT_REVIEWS_MID     = 10   # reviews > 50
WEIGHT_BASE_BONUS      =  5
MAX_SCORE              = 100


class LeadRequest(BaseModel):
    industry: str = Field(..., example="restaurants", description="Type of business to search for")
    location: str = Field(..., example="New York",    description="City or area to search in")
    max_results: int = Field(5, ge=1, le=20,          description="Number of leads to return (1–20)")


class LeadResponse(BaseModel):
    business_name: str
    category:      str
    address:       str
    rating:        float
    reviews:       int
    website:       Optional[str]
    phone:         Optional[str]
    email:         Optional[str]
    lead_score:    int
    lead_type:     str  # HOT | WARM | COLD


JUNK_DOMAINS = {
    "sentry.io", "wixpress.com", "sentry.wixpress.com",
    "example.com", "test.com", "domain.com", "yoursite.com",
    "mailchimp.com", "sendgrid.net", "amazonses.com",
    "noreply.com", "no-reply.com", "bounce.com",
    "w3.org", "schema.org", "google.com", "googleapis.com",
    "cloudflare.com", "wordpress.com", "squarespace.com",
    "godaddy.com", "wix.com", "shopify.com",
}

JUNK_LOCAL_PARTS = {
    "noreply", "no-reply", "donotreply", "do-not-reply",
    "mailer", "bounce", "postmaster", "webmaster",
    "example", "test", "dummy", "placeholder",
    "unsubscribe", "notification", "alert",
    "wordpress", "woocommerce", "shopify",
}

JUNK_PATTERNS = [
    re.compile(r"^[a-f0-9]{16,}@"),
    re.compile(r"^\d{6,}@"),
    re.compile(r"@.*\.(cdn|tracking|mail\d+)\.", re.I),
]

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")


def _is_junk_email(email: str) -> bool:
    email = email.lower().strip()
    local, _, domain = email.partition("@")
    if domain in JUNK_DOMAINS:
        return True
    if any(domain.endswith("." + jd) for jd in JUNK_DOMAINS):
        return True
    if any(junk in local for junk in JUNK_LOCAL_PARTS):
        return True
    if any(p.search(email) for p in JUNK_PATTERNS):
        return True
    return False


def _domain_of(url: str) -> str:
    url = re.sub(r"^https?://", "", url, flags=re.I)
    return url.split("/")[0].lower().lstrip("www.")


# Email Scraper
def extract_email_from_website(url: Optional[str]) -> Optional[str]:
    """Scrape a business website and return the best contact email, or None."""
    if not url:
        return None
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        resp.raise_for_status()
        html = resp.text
    except Exception:
        return None

    raw     = list(set(EMAIL_REGEX.findall(html)))
    clean   = [e for e in raw if not _is_junk_email(e)]
    if not clean:
        return None

    site_domain = _domain_of(url)
    own = [e for e in clean if e.lower().endswith("@" + site_domain)]
    return own[0] if own else clean[0]


# Lead Scoring
def calculate_lead_score(
    has_website: bool,
    has_phone:   bool,
    has_email:   bool,
    rating:      float = 0.0,
    reviews:     int   = 0,
) -> int:
    score  = WEIGHT_BASE_BONUS
    score += WEIGHT_WEBSITE_PRESENT if has_website else WEIGHT_WEBSITE_ABSENT
    if has_phone:  score += WEIGHT_PHONE
    if has_email:  score += WEIGHT_EMAIL
    if   rating >= 4.5: score += WEIGHT_RATING_HIGH
    elif rating >= 4.0: score += WEIGHT_RATING_MID
    if   reviews > 100: score += WEIGHT_REVIEWS_HIGH
    elif reviews > 50:  score += WEIGHT_REVIEWS_MID
    return min(score, MAX_SCORE)


def get_lead_type(score: int) -> str:
    if score >= SCORE_HOT:  return "HOT"
    if score >= SCORE_WARM: return "WARM"
    return "COLD"


# SerpAPI Helpers
def _build_lead(place: dict) -> LeadResponse:
    rating  = float(place.get("rating")  or 0)
    reviews = int(place.get("reviews")   or 0)
    website = place.get("website")
    phone   = place.get("phone")
    email   = extract_email_from_website(website)
    score   = calculate_lead_score(
        has_website=bool(website),
        has_phone=bool(phone),
        has_email=bool(email),
        rating=rating,
        reviews=reviews,
    )
    return LeadResponse(
        business_name=place.get("title",   "Unknown"),
        category=     place.get("type",    "N/A"),
        address=      place.get("address", "N/A"),
        rating=rating,
        reviews=reviews,
        website=website,
        phone=phone,
        email=email,
        lead_score=score,
        lead_type=get_lead_type(score),
    )


def fetch_leads(industry: str, location: str, max_results: int) -> list[LeadResponse]:
    """Core logic: query SerpAPI and return scored LeadResponse objects."""
    if not SERPAPI_KEY:
        raise EnvironmentError("SERP_API_KEY environment variable is not set.")

    query = f"{industry} in {location}"
    logger.info("Searching: %r", query)

    resp = requests.get(
        SERPAPI_URL,
        params={"engine": "google_maps", "q": query, "type": "search", "api_key": SERPAPI_KEY},
        timeout=REQUEST_TIMEOUT,
    )
    resp.raise_for_status()

    places = resp.json().get("local_results", [])
    return [_build_lead(p) for p in places[:max_results]]


# FastAPI Route
@app.post(
    "/enrich-lead",
    response_model=list[LeadResponse],
    summary="Enrich & score business leads",
    response_description="List of scored leads with contact details",
)
def enrich_lead(body: LeadRequest):
    """
    Search Google Maps for businesses matching **industry** in **location**,
    scrape their websites for contact emails, and return scored leads.

    Lead types:
    - **HOT**  → score ≥ 80
    - **WARM** → score ≥ 50
    - **COLD** → score < 50
    """
    try:
        leads = fetch_leads(body.industry, body.location, body.max_results)
        return leads
    except EnvironmentError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except requests.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"SerpAPI error: {e}")
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="SerpAPI request timed out.")
    except Exception as e:
        logger.exception("Unexpected error in /enrich-lead")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("lead_finder:app", host="0.0.0.0", port=8000, reload=True)