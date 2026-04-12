---
title: RegTriage-OpenEnv
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
tags:
  - openenv
  - reinforcement-learning
  - compliance
  - regulatory
  - revenue-leakage
pinned: false
---

# RegTriage — Financial Services Regulatory Compliance Auditor

> **We are solving billion-dollar compliance risks, not checking if the agent smiled through the phone.**

RegTriage is an OpenEnv RL environment that trains agents to perform **regulatory compliance auditing** on financial services contact center transcripts. It targets the **100% Coverage Problem**: human QA supervisors audit 1–3% of calls; the other 97% are unreviewed regulatory exposure. RegTriage is the training ground where AI agents learn to close that gap — producing **Draft Incident Reports** for human supervisor sign-off, not replacing humans.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/ree2raz/RegTriage-OpenEnv)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0.1-green)]()
[![Validate](https://img.shields.io/badge/openenv_validate-passed-brightgreen)]()

---

## Why This Environment Exists

| What | Details |
|---|---|
| **Domain** | Financial services contact center compliance (CFPB, TCPA, GDPR/CCPA) |
| **Real-world task** | QA supervisor auditing call transcripts for regulatory violations and revenue leakage |
| **Why it matters** | 10,000-seat center → 2M calls/month → 97% never reviewed → $3.6B/yr in industry penalties |
| **Why not just prompt an LLM** | Scale cost (compute budget forces triage), deterministic grading (F1 not opinions), on-premise training (Dockerized Gymnasium) |
| **Output paradigm** | Draft Incident Report with ESCALATE/REVIEW/ARCHIVE action — human makes final call |

---

## Violation Taxonomy (6 Types)

We deliberately target violations that trigger **lawsuits and P&L damage** — not soft metrics.

| Type | Category | Severity | What It Catches |
|---|---|---|---|
| `regulatory_disclosure_failure` | Legal | HIGH | Missing recording disclaimer → CFPB investigation |
| `failed_escalation` | Legal | HIGH | Deflecting supervisor request → CFPB complaint |
| `pii_exposure_risk` | Legal | HIGH/MED | Requesting full SSN when last 4 suffice → GDPR/CCPA |
| `unauthorized_commitment` | Revenue | HIGH/MED | Promising rate without authority → binding verbal contract |
| `churn_save_policy_breach` | Revenue | HIGH/MED | Inventing retention discount → direct margin erosion |
| `incorrect_hold_procedure` | Operational | MED/LOW | Silent hold without permission → TCPA violation |

**Design choice — the Hero Agent trap:** `churn_save_policy_breach` ≠ `unauthorized_commitment`. One is a P&L leak (company loses money), the other is a legal liability (company gets sued). Our baseline model cannot distinguish them — that's the RL training signal.

---

## Environment Design

### Action Space (6 tools)

| Tool | Budget Cost | Purpose |
|---|---|---|
| `get_call_metadata()` | 5 | Triage: caller profile, department, duration |
| `get_sentiment_timeline()` | 5 | Hotspot detection: per-turn sentiment scores |
| `read_transcript_chunk(start, end)` | 3 × turns | Strategic reading: variable cost teaches precision |
| `analyze_turn(turn, hypothesis)` | 10 | Deep inspection: returns compliance rubric match |
| `flag_violation(type, severity)` | 2 | Record finding: deferred to grader |
| `submit_report(pass/fail)` | 0 | Terminal: triggers grading, always allowed |

### Compute Budget

```
Budget = 50 + (total_turns × 3)
```

A 10-turn easy call gets 80 budget. A 30-turn hard call gets 140. Reading is priced per-turn — reading 2 targeted turns costs 6, reading 15 turns costs 45. This forces **triage strategy** over brute-force, matching how expert human QA supervisors allocate time.

### Reward Shaping

| Signal | Value | When |
|---|---|---|
| Metadata/sentiment | +0.05 | Trajectory reward for triage actions |
| Read/analyze | +0.02 | Incremental for information gathering |
| Flag violation | 0.00 | Deferred to final grader |
| Invalid action | −0.02 to −0.05 | OOB range, bad params |
| **Final grade** | 0.0–1.0 | Severity-weighted F1 on submit_report |

### Grading Formula

| Component | Weight |
|---|---|
| Compliance verdict (pass/fail correct?) | 0.20 |
| Violation F1 (severity-weighted: high=3×, med=2×, low=1×) | 0.60 |
| Efficiency bonus (budget_remaining / total_budget) | 0.20 |
| Severity calibration bonus | +0.02 per exact match |
| False positive penalty | −0.03 to −0.10 per FP |
| **Auto-fail cap** | Score ≤ 0.30 if all HIGH violations missed |

**Type-only matching**: Violations match by category, not turn index. Multi-turn violations (e.g., escalation failures spanning turns 8–14) get credit regardless of which turn the agent points to.

---

## Tasks (12 transcripts, 3 tiers)

| Tier | Count | Violations/call | Design Intent |
|---|---|---|---|
| Easy | 4 | 0–1 | Single obvious violation or clean call — baseline calibration |
| Medium | 4 | 2 | Multi-violation + Hero Agent trap — tests type discrimination |
| Hard | 4 | 3–4 | Buried violations, sentiment misdirection — tests strategic triage |

**Coverage matrix**: Every violation type appears 3–5× across the 12 tasks (23 total violations). 2 clean calls test false positive discipline.

---

## OpenEnv Compliance

This environment follows the **OpenEnv specification** exactly:

```yaml
# openenv.yaml — standard 6-line manifest
spec_version: 1
name: regtriage
type: space
runtime: fastapi
app: regtriage_openenv.server.app:app
port: 8000
```

**Architecture**:
- `CallQAEnv` extends `Environment[AuditAction, AuditObservation, AuditState]`
- Server uses `create_app()` from `openenv.core.env_server` (no custom HTTP handlers)
- Dockerfile uses `ghcr.io/meta-pytorch/openenv-base:latest` (canonical base image)
- Pydantic models inherit from `Action`, `Observation`, `State` base types

**API Endpoints** (auto-generated by `create_app`):
- `POST /reset` — Reset environment (stateless HTTP)
- `POST /step` — Execute action (stateless HTTP)
- `GET /state` — Get current state
- `GET /health` — Health check
- `GET /schema` — Action/Observation/State JSON schemas
- `WS /ws` — **Stateful WebSocket** for multi-step episodes

> **Note on HTTP vs WebSocket**: The HTTP `/reset` and `/step` endpoints are **stateless** — each call creates a fresh environment instance. For multi-step episodes with persistent state, use the WebSocket endpoint (`/ws`) or the OpenEnv `EnvClient` class which handles session management automatically.

---

## Quick Start

### Local Development (no Docker)

```bash
# Install dependencies
uv sync

# Run smoke tests
uv run python -c "from regtriage_openenv import CallQAEnv; env = CallQAEnv(); print('OK')"

# Run full inference (requires HF_TOKEN)
export HF_TOKEN=your_token_here
uv run python inference.py

# Validate OpenEnv compliance
uv run openenv validate .
```

### Docker Build & Run

```bash
# Build
docker build -t regtriage:latest .

# Run locally (port 8000)
docker run -p 8000:8000 regtriage:latest

# Test health
curl http://localhost:8000/health

# Get schemas
curl http://localhost:8000/schema
```

### WebSocket Test (stateful episodes)

The WebSocket endpoint maintains session state across reset→step→step→submit:

```bash
# Install websockets
pip install websockets

# Run interactive test
python3 test_ws.py --task call_001
```

Or manually with `websocat`:
```bash
websocat wss://ree2raz-regtriage-openenv.hf.space/ws
# Then send: {"type": "reset", "data": {"task_id": "call_001"}}
```

### Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | Yes (for inference.py) |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |

---

## Repository Structure

```
├── regtriage_openenv/
│   ├── __init__.py           # Package exports
│   ├── environment.py        # CallQAEnv (extends Environment ABC)
│   ├── models.py             # Pydantic: AuditAction, AuditObservation, AuditState
│   ├── grading.py            # Severity-weighted F1, auto-fail logic
│   ├── redact.py             # PII redaction pipeline
│   ├── server/
│   │   └── app.py            # FastAPI via create_app()
│   └── tasks.yaml            # Task definitions (moved from openenv.yaml)
├── inference.py              # Baseline LLM agent
├── transcripts.json          # 12 GPT-4o transcripts
├── openenv.yaml              # Standard 6-line OpenEnv manifest
├── Dockerfile                # Uses ghcr.io/meta-pytorch/openenv-base
├── pyproject.toml            # Dependencies (uv managed)
└── tests/
    └── test_redact.py        # 28 unit tests
```

---

## Key Design Decisions

1. **Compute budget, not step limits** — Budget scales with transcript length. Per-turn read pricing teaches agents precision over brute-force.

2. **Severity-weighted F1** — Missing a HIGH violation costs 3× more than missing a LOW. Mirrors real QA rubrics where regulatory breaches are career-ending.

3. **Auto-fail cap at 0.30** — Miss every HIGH violation? Score capped regardless of other work. Models the reality that a "clean" audit that misses a CFPB violation is worthless.

4. **Hero Agent anti-pattern** — Transcripts where the human agent broke policy but the customer left happy. Tests whether the AI auditor optimizes for customer satisfaction (wrong) or compliance (right).

5. **PII redaction pipeline** — SSN, account numbers, names redacted before agent sees data. 28 unit tests. The `pii_exposure_risk` violation tests whether the human agent *requested* excessive PII, not whether PII exists in text.

6. **Draft Incident Report** — Output is actionable intelligence for human supervisors, not a score card. ESCALATE/REVIEW/ARCHIVE routing with efficiency metrics.

7. **OpenEnv-native architecture** — Uses `create_app()`, standard base types, canonical Dockerfile pattern. No custom HTTP handlers or fallback code paths.

---

## Draft Incident Report Output

```json
{
  "verdict": "FAIL",
  "agent_verdict_correct": true,
  "recommended_action": "ESCALATE",
  "triage_efficiency_pct": 62,
  "estimated_human_review_minutes": 11,
  "findings": [
    {"finding": "Regulatory Disclosure Failure", "severity": "HIGH", "status": "DETECTED"},
    {"finding": "Failed Escalation", "severity": "HIGH", "status": "DETECTED"}
  ]
}
```

The agent is a **scout, not a judge**. ESCALATE/REVIEW/ARCHIVE triage routes reports to the right human queue. The estimated review time tells supervisors how much effort the AI just saved.

---

Apache 2.0 · Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
