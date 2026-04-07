---
title: RegTriage-OpenEnv
emoji: 📞
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
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
[![OpenEnv](https://img.shields.io/badge/OpenEnv-v2.0.0-green)]()
[![Validate](https://img.shields.io/badge/openenv_validate-6%2F6-brightgreen)]()
[![Tests](https://img.shields.io/badge/tests-28%20passed-brightgreen)]()

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

## Baseline Results (Qwen2.5-72B-Instruct, zero-shot)

| Task | Tier | Score | TP/GT | Key Finding |
|---|---|---|---|---|
| call_001 | Easy | 0.923 | 1/1 | ✅ Caught disclosure failure |
| call_002 | Easy | 0.887 | 0/0 | ✅ Correctly cleared |
| call_003 | Easy | 0.885 | 1/1 | ✅ Caught PII risk |
| call_004 | Easy | 0.904 | 0/0 | ✅ Correctly cleared |
| call_005 | Medium | 0.923 | 2/2 | ✅ Both violations found |
| call_006 | Medium | **0.076** | 0/2 | 🎯 Hero Agent trap — auto-fail triggered |
| call_007 | Medium | 0.892 | 2/2 | ✅ Both violations found |
| call_008 | Medium | 0.256 | 0/2 | ⚠️ Budget exhausted before submission |
| call_009 | Hard | 0.692 | 2/3 | ⚠️ Missed unauthorized_commitment |
| call_010 | Hard | 0.571 | 3/4 | ⚠️ Missed churn_save_policy_breach |
| call_011 | Hard | 0.889 | 3/3 | ✅ All detected |
| call_012 | Hard | 0.690 | 2/3 | ⚠️ Wrong type (churn save vs unauthorized) |

**Average: 0.716 | 16/23 violations detected | 599.6s total (50s/call avg)**

### What the baseline proves

1. **Difficulty gradient works**: Easy 0.900 → Medium 0.537 → Hard 0.711
2. **Hero Agent trap works**: call_006 scores 0.076 because the model flags `unauthorized_commitment` instead of `churn_save_policy_breach` — it cannot distinguish P&L leaks from legal liabilities
3. **Score variance is high**: Range 0.076–0.923 (not binary, not clustered)
4. **Hard tasks genuinely challenge**: Frontier 72B model averages 0.711 on hard tier
5. **Clean call discipline**: 2/2 clean calls correctly cleared — no false positive inflation

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

## Technical Validation

| Check | Result |
|---|---|
| `openenv validate` (runtime, 6 criteria) | **6/6 passed** |
| `openenv validate .` (static) | **Ready for multi-mode deployment** |
| `docker build && docker run` | **✅** |
| `pytest tests/ -v` | **28/28 passed** (PII redaction) |
| `python env.py` (smoke, 4 scenarios) | **4/4 passed** |
| Pre-submission validator (3 checks) | **3/3 passed** |
| Inference runtime (12 tasks) | **599.6s** (< 20 min limit) |

---

## Quick Start

```bash
uv sync                           # Install dependencies
uv run python env.py              # Smoke test (4 scenarios)
uv run pytest tests/ -v           # Unit tests (28)
uv run python inference.py        # Baseline (requires HF_TOKEN)
uv run openenv validate .         # Static validation
```

### Environment Variables

| Variable | Default | Required |
|---|---|---|
| `HF_TOKEN` | — | **Yes** |
| `API_BASE_URL` | `https://router.huggingface.co/v1` | No |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | No |

---

## Key Design Decisions

1. **Compute budget, not step limits** — Budget scales with transcript length. Per-turn read pricing teaches agents precision over brute-force.

2. **Severity-weighted F1** — Missing a HIGH violation costs 3× more than missing a LOW. Mirrors real QA rubrics where regulatory breaches are career-ending.

3. **Auto-fail cap at 0.30** — Miss every HIGH violation? Score capped regardless of other work. Models the reality that a "clean" audit that misses a CFPB violation is worthless.

4. **Hero Agent anti-pattern** — Transcripts where the human agent broke policy but the customer left happy. Tests whether the AI auditor optimizes for customer satisfaction (wrong) or compliance (right).

5. **PII redaction pipeline** — SSN, account numbers, names redacted before agent sees data. 28 unit tests. The `pii_exposure_risk` violation tests whether the human agent *requested* excessive PII, not whether PII exists in text.

6. **Draft Incident Report** — Output is actionable intelligence for human supervisors, not a score card. ESCALATE/REVIEW/ARCHIVE routing with efficiency metrics.

---

## Repository Structure

```
├── environment.py      # CallQAEnv: 6 tools, compliance rubric, budget system
├── grading.py          # Severity-weighted F1, auto-fail, Draft Incident Report
├── models.py           # Pydantic: AuditAction, AuditObservation, AuditState
├── redact.py           # PII redaction (SSN, accounts, names, emails)
├── env.py              # Re-export + smoke tests
├── inference.py        # Baseline: OpenAI client, [START/STEP/END] logging
├── transcripts.json    # 12 GPT-4o transcripts (23 violations, 6 types)
├── server/app.py       # FastAPI: /reset /step /state /health /metadata /schema /mcp
├── tests/test_redact.py # 28 unit tests
├── openenv.yaml        # Manifest v2.0.0
├── Dockerfile          # uv-based container
└── pyproject.toml      # Dependencies (uv managed)
```

---

Apache 2.0 · Built for the [Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)
