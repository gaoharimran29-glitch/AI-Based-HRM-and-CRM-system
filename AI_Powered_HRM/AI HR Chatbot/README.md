# AI HR Chatbot — Internal Employee Assistant

An AI-powered HR assistant that lets employees query their own leave balance,
salary breakdown, and HR policies through natural language.

## Architecture

```
User message
     │
     ▼
┌─────────────┐
│  Guardrails │  ← length check, rate limit, prompt injection,
│  (guardrails│    cross-user ID detection, abuse filter
│   .py)      │
└──────┬──────┘
       │ passes
       ▼
┌─────────────┐
│  LLM Agent  │  ← Gemini 1.5 Flash via LangChain
│ (llm_layer  │    ReAct agent with tool calling
│   .py)      │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Tools    │  ← leave_balance_tool, salary_tool,
│  (tools.py) │    employee_info_tool, policy_tool
└──────┬──────┘
       │
       ▼
┌─────────────┐
│   MongoDB   │  ← employees, leaves, salary, policies
└─────────────┘
       │
       ▼
┌─────────────┐
│ Audit Logger│  ← every query/block/error logged to hr_audit.log
└─────────────┘
```

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
Copy `.env.example` to `.env` and fill in your key:
```bash
cp .env.example .env
```

`.env` contents:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

Get your free key at: https://aistudio.google.com/app/apikey

### 3. Start MongoDB
```bash
mongod
```

### 4. Seed the database
```bash
python feed_data.py
```

### 5. Run the chatbot
```bash
python llm_layer.py
```

## Files

| File | Purpose |
|------|---------|
| `llm_layer.py` | Main chatbot — LLM agent + chat loop |
| `tools.py` | Tool functions that fetch from MongoDB |
| `guardrails.py` | All security/validation checks |
| `audit_logger.py` | Structured JSON audit logging |
| `feed_data.py` | Seeds MongoDB with dummy HR data |
| `test.py` | Tests DB tools directly (no API key needed) |
| `test_guardrails.py` | Tests all guardrail checks (no API key needed) |

## Security Features

- **Prompt injection protection** — detects and blocks attempts to override instructions
- **Cross-user data isolation** — users cannot reference other employees' IDs
- **Rate limiting** — max 10 messages per minute per employee
- **Input length limit** — max 500 characters per message
- **Abuse filter** — blocks profanity and hostile messages
- **Audit log** — every interaction logged to `hr_audit.log` as JSON
- **Scoped tools** — employee_id is baked into tools at agent creation time, not user-supplied

## Integration Note

The `employee_id` in `llm_layer.py` is currently hardcoded for local testing:
```python
LOGGED_IN_EMPLOYEE_ID = "emp_001"
```
In production, replace this with the ID extracted from your auth system
(JWT token, session, etc.) by the backend team. The AI layer is already
designed to accept any employee_id — authentication is a backend concern.

## Sample Queries

```
How many leaves do I have?
What is my salary breakdown?
What is the maternity leave policy?
Tell me my profile details
```