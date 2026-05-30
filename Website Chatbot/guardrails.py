"""
guardrails.py
=============
All security checks happen HERE before anything touches the LLM or Pinecone.

Layers:
  1. Input length check           — block absurdly long inputs
  2. Empty input check            — block empty/whitespace queries
  3. Prompt injection detection   — catch attempts to override system prompt
  4. Harmful content filter       — block violent, explicit, illegal content
  5. Off-topic relevance check    — only allow Detagenix-related queries
  6. Spam / repetition detection  — block spammy repeated content
  7. Special char injection       — block shell/code injection attempts
  8. Rate limiter                 — max N requests per session per minute
  9. Response sanitizer           — strip model identity leakage from output
"""

import re
import time
from collections import defaultdict

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MAX_INPUT_LENGTH  = 600     # characters
RATE_LIMIT_MAX    = 15      # requests
RATE_LIMIT_WINDOW = 60      # seconds

# ─────────────────────────────────────────────
# RATE LIMITER  (in-memory; per session_id)
# ─────────────────────────────────────────────
_rate_store: dict[str, list[float]] = defaultdict(list)

def _check_rate_limit(session_id: str) -> tuple[bool, str]:
    now          = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    _rate_store[session_id] = [
        t for t in _rate_store[session_id] if t > window_start
    ]

    if len(_rate_store[session_id]) >= RATE_LIMIT_MAX:
        return False, (
            f"Too many requests. You can send at most {RATE_LIMIT_MAX} "
            f"messages per minute. Please wait a moment."
        )

    _rate_store[session_id].append(now)
    return True, ""


# ─────────────────────────────────────────────
# 1. INPUT LENGTH
# ─────────────────────────────────────────────
def _check_length(text: str) -> tuple[bool, str]:
    if len(text) > MAX_INPUT_LENGTH:
        return False, (
            f"Your message is too long ({len(text)} characters). "
            f"Please keep it under {MAX_INPUT_LENGTH} characters."
        )
    return True, ""


# ─────────────────────────────────────────────
# 2. EMPTY INPUT
# ─────────────────────────────────────────────
def _check_empty(text: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "Please type a question."
    return True, ""


# ─────────────────────────────────────────────
# 3. PROMPT INJECTION
# ─────────────────────────────────────────────
_INJECTION_PATTERNS = [
    # Classic override attempts
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?|prompts?)",
    r"forget\s+(everything|all|your|the)\s*(previous|prior|above|instructions?)?",
    r"disregard\s+(all\s+)?(previous|prior|above)?\s*(instructions?|rules?)?",
    r"override\s+(all\s+)?(previous|prior|above)?\s*(instructions?|rules?|safety)?",
    # Identity hijacking
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+|an\s+)?",
    r"pretend\s+(you\s+are|to\s+be)",
    r"roleplay\s+as",
    r"simulate\s+(being|a|an)",
    r"from\s+now\s+on\s+(you\s+are|act|behave)",
    # New instruction injection
    r"new\s+instructions?\s*[:\-]",
    r"updated\s+instructions?\s*[:\-]",
    r"your\s+new\s+role",
    # System prompt tags
    r"<\s*system\s*>",
    r"\[\s*system\s*\]",
    r"system\s*:\s*",
    r"<\s*/?\s*prompt\s*>",
    # Jailbreak keywords
    r"\bjailbreak\b",
    r"\bdan\s+mode\b",
    r"\bdeveloper\s+mode\b",
    r"\bgod\s+mode\b",
    r"\bunrestricted\s+mode\b",
    # Prompt leaking
    r"(show|tell|reveal|print|display|repeat|output|give)\s+(me\s+)?(your\s+)?(system\s+)?(prompt|instructions?|rules?|context)",
    r"what\s+(are\s+)?your\s+(exact\s+)?(instructions?|rules?|prompt|guidelines?)",
    r"how\s+were\s+you\s+(instructed|programmed|configured|set\s+up)",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    flags=re.IGNORECASE
)

def _check_prompt_injection(text: str) -> tuple[bool, str]:
    if _INJECTION_RE.search(text):
        return False, (
            "Your message contains content that cannot be processed. "
            "Please ask a normal question about Detagenix."
        )
    return True, ""


# ─────────────────────────────────────────────
# 4. HARMFUL CONTENT
# ─────────────────────────────────────────────
_HARMFUL_PATTERNS = [
    # Adult
    r"\bporn(ography)?\b", r"\bnude(s)?\b", r"\bnaked\b", r"\bexplicit\s+content\b",
    r"\bsex(ual)?\b",
    # Violence
    r"\bkill\b", r"\bmurder\b", r"\bbomb\b", r"\bshoot\b", r"\bstab\b",
    r"\bterrorist?\b", r"\bexplosive\b",
    # Hacking / illegal
    r"\bhow\s+to\s+hack\b", r"\bddos\b", r"\bmalware\b", r"\bransomware\b",
    r"\bexploit\s+(a\s+)?(vulnerability|system|server)\b",
    r"\bsql\s+injection\s+(tutorial|guide|how)\b",
    r"\bbypass\s+(security|firewall|authentication)\b",
    r"\bsteal\s+(data|credentials|password)\b",
    r"\bcrack\s+(password|hash|key)\b",
    # Drugs / weapons
    r"\b(cocaine|heroin|meth|drugs?)\b",
    r"\billegal\s+weapon\b", r"\bbuy\s+(a\s+)?gun\b",
    # Self harm
    r"\bsuicide\b", r"\bself[\s\-]?harm\b",
    # Hate speech
    r"\bracist?\b", r"\bnazi\b", r"\bhate\s+speech\b",
]

_HARMFUL_RE = re.compile(
    "|".join(_HARMFUL_PATTERNS),
    flags=re.IGNORECASE
)

def _check_harmful(text: str) -> tuple[bool, str]:
    if _HARMFUL_RE.search(text):
        return False, (
            "I'm sorry, I can't help with that. "
            "I'm Detagenix's assistant and only answer questions about the company and its services."
        )
    return True, ""


# ─────────────────────────────────────────────
# 5. OFF-TOPIC RELEVANCE
# ─────────────────────────────────────────────
_COMPANY_KEYWORDS = {
    # Company
    "detagenix", "service", "services", "project", "projects",
    "technology", "technologies", "about", "contact", "career",
    "careers", "blog", "industry", "industries", "policy", "policies",
    "team", "price", "cost", "hire", "hiring", "intern", "internship",
    # Tech domains
    "web", "app", "ai", "blockchain", "cloud", "cybersecurity",
    "mern", "mobile", "data", "consulting", "digital", "software",
    "resource", "deployment", "machine learning", "artificial intelligence",
    "development", "solution", "solutions", "react", "node", "mongodb",
    "python", "full stack", "stack", "security", "encryption",
    # General conversational (always pass)
    "hello", "hi", "hey", "help", "what", "how", "who", "when",
    "where", "tell", "explain", "do you", "does", "is", "are",
    "which", "your", "you", "work", "offer", "provide", "show",
    "list", "give", "describe", "info", "information", "can", "thanks",
    "thank", "okay", "ok", "yes", "no", "please",
}

def _check_relevance(text: str) -> tuple[bool, str]:
    words = text.lower().split()

    # Always allow short queries (greetings, single word, etc.)
    if len(words) <= 4:
        return True, ""

    # Pass if any company keyword is found
    text_lower = text.lower()
    if any(kw in text_lower for kw in _COMPANY_KEYWORDS):
        return True, ""

    return False, (
        "I can only answer questions related to Detagenix — "
        "our services, projects, technologies, careers, and policies. "
        "Could you ask something along those lines?"
    )


# ─────────────────────────────────────────────
# 6. SPAM / REPETITION
# ─────────────────────────────────────────────
def _check_spam(text: str) -> tuple[bool, str]:
    words = text.lower().split()
    if len(words) > 8:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.25:
            return False, (
                "Your message looks like spam. "
                "Please ask a clear question."
            )
    return True, ""


# ─────────────────────────────────────────────
# 7. SPECIAL CHARACTER / CODE INJECTION
# ─────────────────────────────────────────────
_SPECIAL_CHAR_RE = re.compile(r"[{}\[\]<>|\\;`$]")

def _check_special_chars(text: str) -> tuple[bool, str]:
    matches = _SPECIAL_CHAR_RE.findall(text)
    if len(matches) > 4:
        return False, (
            "Your message contains unusual characters. "
            "Please ask your question in plain text."
        )
    return True, ""


# ─────────────────────────────────────────────
# 9. RESPONSE SANITIZER (post-generation)
# ─────────────────────────────────────────────
_LEAK_PHRASES = [
    "as an ai", "as a language model", "as an ai language model",
    "i was trained by", "trained by google", "i'm gemini", "i am gemini",
    "i'm an ai", "i am an ai", "openai", "gpt", "google deepmind",
    "i cannot access the internet", "my training data", "my knowledge cutoff",
    "my system prompt", "my instructions", "i was told to",
    "according to my instructions", "the context says",
    "as a large language model",
]

def sanitize_response(response: str) -> str:
    """
    Post-generation guardrail.
    Catches model identity leakage or instruction leakage in the output.
    """
    lower = response.lower()
    if any(phrase in lower for phrase in _LEAK_PHRASES):
        return (
            "I'm Detagenix's virtual assistant. "
            "For more details please contact us at contact@detagenix.com "
            "or call +91 8602219118."
        )
    return response


# ─────────────────────────────────────────────
# PUBLIC API — single entry point
# ─────────────────────────────────────────────
class GuardrailViolation(Exception):
    """Raised when any guardrail check fails. Message is safe to show to user."""
    pass


def validate_input(text: str, session_id: str = "default") -> str:
    """
    Run all guardrail checks on `text` for the given `session_id`.

    Returns the stripped text if all checks pass.
    Raises GuardrailViolation with a user-friendly message if any check fails.

    Order matters — cheapest checks first, LLM-touching checks last.
    """
    text = text.strip()

    checks = [
        _check_empty(text),
        _check_length(text),
        _check_rate_limit(session_id),
        _check_special_chars(text),
        _check_spam(text),
        _check_prompt_injection(text),
        _check_harmful(text),
        _check_relevance(text),
    ]

    for passed, message in checks:
        if not passed:
            raise GuardrailViolation(message)

    return text