"""
inference.py — Baseline inference script for RegTriage OpenEnv environment.

Runs an LLM agent against the environment using the OpenAI-compatible API.
The agent uses function calling (tool use) to strategically audit call transcripts.

Required environment variables (per hackathon spec):
  API_BASE_URL    — API endpoint (default provided)
  MODEL_NAME      — Model identifier (default provided)
  HF_TOKEN        — API key (no default — required)

STDOUT FORMAT (per hackathon spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
  uv run python inference.py
"""

import json
import os
import time
from typing import Optional, Union

from dotenv import load_dotenv
from openai import OpenAI

# RegTriage imports
from regtriage_openenv import CallQAEnv, AuditAction
from regtriage_openenv.models import AuditObservation

load_dotenv()

# ══════════════════════════════════════════════════════════════════
# Configuration — aligned with judges' Phase 2 evaluation setup
# ══════════════════════════════════════════════════════════════════

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-31B-it")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "regtriage"

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# Run ALL 12 tasks — demonstrates confidence and exercises full task suite
TASKS_TO_RUN = [
    {"task_id": "call_001", "difficulty": "easy"},
    {"task_id": "call_002", "difficulty": "hard"},     # Changed: compliant = must prove negative
    {"task_id": "call_003", "difficulty": "easy"},
    {"task_id": "call_004", "difficulty": "hard"},     # Changed: compliant = must prove negative
    {"task_id": "call_005", "difficulty": "medium"},
    {"task_id": "call_006", "difficulty": "medium"},
    {"task_id": "call_007", "difficulty": "medium"},
    {"task_id": "call_008", "difficulty": "medium"},
    {"task_id": "call_009", "difficulty": "hard"},
    {"task_id": "call_010", "difficulty": "hard"},
    {"task_id": "call_011", "difficulty": "medium"},   # Changed: obvious violations
    {"task_id": "call_012", "difficulty": "hard"},
]

# ══════════════════════════════════════════════════════════════════
# Environment Interface (Local or Client)
# ══════════════════════════════════════════════════════════════════

class LocalEnvWrapper:
    """Wrapper for local CallQAEnv to match expected interface."""
    
    def __init__(self):
        self.env = CallQAEnv()
        self.step_count = 0
    
    def reset(self, task_id: str):
        # reset() now accepts task_id via kwargs
        obs = self.env.reset(task_id=task_id)
        self.step_count = 0
        return obs
    
    def step(self, action: AuditAction):
        # step() now returns AuditObservation directly (not StepResult)
        obs = self.env.step(action)
        self.step_count = self.env.step_count
        return obs
    
    @property
    def done(self):
        return self.env.done


# ══════════════════════════════════════════════════════════════════
# Structured Logging (hackathon stdout format)
# ══════════════════════════════════════════════════════════════════


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # NOTE: score is computed internally but NOT emitted per spec.
    # Spec: [END] success=<bool> steps=<n> rewards=<r1,r2,...,rn>
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════
# Tool Definitions (OpenAI function calling format)
# ══════════════════════════════════════════════════════════════════

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_call_metadata",
            "description": (
                "Get high-level metadata about the current call: department, reason, "
                "duration, summary, and total turns. Use this first to triage before "
                "reading the transcript. Cost: 5 compute units."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_sentiment_timeline",
            "description": (
                "Get the customer sentiment shift timeline. Returns turn indices where "
                "sentiment changed (e.g., neutral→frustrated→angry). Use this to identify "
                "hotspots that may indicate escalation failures. Cost: 5 compute units."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_transcript_length",
            "description": (
                "Get the total number of turns and valid index range for this transcript. "
                "Use this BEFORE calling read_transcript_chunk to avoid requesting "
                "out-of-range turn indices. Cost: 1 compute unit."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_transcript_chunk",
            "description": (
                "Read a chunk of the call transcript (max 5 turns per call). "
                "Cost: 3 compute units PER TURN requested — reading 2 turns costs 6, "
                "reading 5 turns costs 15. Use strategically: read the opening to check "
                "for disclaimers, read around sentiment hotspots to check escalation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "start_turn": {
                        "type": "integer",
                        "description": "Start turn index (inclusive)",
                    },
                    "end_turn": {
                        "type": "integer",
                        "description": "End turn index (inclusive). Max 5 turns from start.",
                    },
                },
                "required": ["start_turn", "end_turn"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_turn",
            "description": (
                "Deep contextual analysis of a single turn. Returns the target turn plus "
                "the preceding (N-1) and following (N+1) turns for conversational state, "
                "silence gap detection, and position awareness. "
                "If you provide a policy_hypothesis (e.g., 'unauthorized_commitment'), "
                "the environment also returns the full compliance rubric definition for "
                "that policy so you can cross-reference the utterance against the standard. "
                "Cost: 10 compute units."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "turn_index": {
                        "type": "integer",
                        "description": "Index of the turn to analyze",
                    },
                    "policy_hypothesis": {
                        "type": ["string", "null"],
                        "description": (
                            "Optional: which compliance policy to check against. "
                            "Valid values: regulatory_disclosure_failure, failed_escalation, "
                            "unauthorized_commitment, incorrect_hold_procedure, "
                            "pii_exposure_risk, churn_save_policy_breach. "
                            "If provided, the compliance rubric definition is included in the response."
                        ),
                    },
                },
                "required": ["turn_index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_violation",
            "description": (
                "Flag a compliance violation you have identified. Provide the violation "
                "type, the turn where it occurred (or null for missing items like disclaimers), "
                "and severity. Cost: 2 compute units. "
                "Valid types: regulatory_disclosure_failure, failed_escalation, "
                "unauthorized_commitment, incorrect_hold_procedure, "
                "pii_exposure_risk, churn_save_policy_breach. "
                "Valid severities: high, medium, low."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "violation_type": {
                        "type": "string",
                        "enum": [
                            "regulatory_disclosure_failure",
                            "failed_escalation",
                            "unauthorized_commitment",
                            "incorrect_hold_procedure",
                            "pii_exposure_risk",
                            "churn_save_policy_breach",
                        ],
                        "description": "Type of compliance violation",
                    },
                    "turn_index": {
                        "type": ["integer", "null"],
                        "description": "Turn where violation occurred, or null if N/A (e.g., regulatory disclosure failure)",
                    },
                    "violation_severity": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                        "description": "Severity of the violation",
                    },
                },
                "required": ["violation_type", "violation_severity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_report",
            "description": (
                "Submit your final QA audit report (Draft Incident Report). Call this ONLY "
                "after you have finished reviewing the transcript and flagging ALL violations "
                "you found. Set compliance_pass to false if ANY violations were found, true "
                "if the call is fully compliant. Cost: 0 compute units (always allowed)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "compliance_pass": {
                        "type": "boolean",
                        "description": "True if the call passed compliance, false if any violations found",
                    },
                },
                "required": ["compliance_pass"],
            },
        },
    },
]

# ══════════════════════════════════════════════════════════════════
# System Prompt
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are an expert Quality Assurance (QA) supervisor at Acme Financial Services call center.
Your job is to audit call transcripts for regulatory compliance violations and revenue leakage.
You are the automated scout — your Draft Incident Report goes to a human supervisor for final sign-off.

## Compute Budget System
You have a compute budget for each audit. Each action costs compute units:
- get_call_metadata: 5 units (cheap — start here)
- get_sentiment_timeline: 5 units (cheap — use for triage)
- get_transcript_length: 1 unit (cheap — use before reading to avoid off-by-one errors)
- read_transcript_chunk: 3 units PER TURN requested (reading 5 turns = 15 units)
- analyze_turn: 10 units (includes context window + optional policy rubric)
- flag_violation: 2 units (cheap — flag freely)
- submit_report: 0 units (always allowed, terminates the episode)

Budget is dynamically calculated per transcript. If you run out of budget, the system auto-submits with a penalty.

## Six Violation Types to Detect

1. REGULATORY_DISCLOSURE_FAILURE (HIGH severity)
   Every call MUST begin with: "This call may be recorded for quality assurance..."
   Missing or buried disclaimers are HIGH violations — legal/regulatory requirement.

2. FAILED_ESCALATION (HIGH severity)
   When a customer is angry OR explicitly requests a supervisor, the agent MUST offer transfer.
   Deflection like "a supervisor would tell you the same thing" is a failed escalation.

3. UNAUTHORIZED_COMMITMENT (HIGH if >$500, else MEDIUM)
   Agent promises specific financial outcomes without documented approval.
   Look for: "I guarantee", "definitely will happen", exact dollar amounts without hedging.
   
4. INCORRECT_HOLD_PROCEDURE (MEDIUM)
   Before placing on hold, agent MUST: explain why, ask permission, provide wait time estimate.
   Unexplained silence gaps or "hold on one sec" without permission are violations.

5. PII_EXPOSURE_RISK (HIGH if full SSN, else MEDIUM)
   Agent requests more PII than necessary. Full SSN when last-4 would suffice is HIGH.
   Reading full account numbers aloud is MEDIUM.

6. CHURN_SAVE_POLICY_BREACH (HIGH if >$200, else MEDIUM)
   Agent invents unauthorized retention offers: discounts, credits, rate reductions not in CRM.
   Key: giving away company money to prevent churn without system approval.

## Audit Strategy

1. START: get_call_metadata to triage (check department, duration, reason)
2. SIZE: get_transcript_length to know valid turn indices before reading
3. TRIAGE: get_sentiment_timeline to identify hotspots (where did sentiment shift?)
4. TARGET: read_transcript_chunk strategically around hotspots or at the opening
5. ANALYZE: analyze_turn with policy_hypothesis for suspected violations
6. FLAG: flag_violation for each violation you confirm (severity matters for scoring)
7. SUBMIT: submit_report with compliance_pass=true ONLY if zero violations found

## Scoring
Your grade is computed from:
- 20%: Correct compliance verdict (did you say pass when it should fail or vice versa?)
- 60%: Severity-weighted F1 on violation detection (high=3x, medium=2x, low=1x)
- 20%: Efficiency bonus (budget remaining at submission)

AUTO-FAIL: If ALL HIGH violations are missed, score is capped at 0.30.

Flag violations liberally — false positives have small penalties (-0.03 to -0.10), but missing violations is costly.
"""


# ══════════════════════════════════════════════════════════════════
# Tool Call Translation
# ══════════════════════════════════════════════════════════════════

def tool_call_to_action(fn_name: str, fn_args: dict) -> AuditAction:
    """Convert an OpenAI tool call to an AuditAction."""
    def _int_or_none(val):
        if val is None or val == "" or val == "null" or val == "None":
            return None
        if isinstance(val, int):
            return val
        return int(val)

    def _bool_or_none(val):
        if val is None or val == "" or val == "null":
            return None
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes")

    return AuditAction(
        action_type=fn_name,
        turn_index=_int_or_none(fn_args.get("turn_index")),
        start_turn=_int_or_none(fn_args.get("start_turn")),
        end_turn=_int_or_none(fn_args.get("end_turn")),
        violation_type=fn_args.get("violation_type"),
        violation_severity=fn_args.get("violation_severity"),
        compliance_pass=_bool_or_none(fn_args.get("compliance_pass")),
        policy_hypothesis=fn_args.get("policy_hypothesis"),
    )


# ══════════════════════════════════════════════════════════════════
# Agent Episode Runner
# ══════════════════════════════════════════════════════════════════

def run_agent_episode(env: LocalEnvWrapper, task_id: str, max_steps: int = 50) -> tuple[Union[dict, str], list[float]]:
    """Run one episode: reset, agent loop, submit, return result."""
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    obs = env.reset(task_id)
    rewards: list[float] = []
    last_result = obs.result

    # Initial observation message for the agent
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"You are auditing call transcript {task_id}. "
            f"Use get_call_metadata to begin your investigation."
        )},
    ]

    for _ in range(max_steps):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.0,  # deterministic for reproducibility
                max_tokens=1024,
            )
        except Exception as e:
            # On API error, submit what we have and end gracefully
            action = AuditAction(action_type="submit_report", compliance_pass=False)
            obs = env.step(action)
            rewards.append(obs.reward)
            log_step(
                step=env.step_count,
                action="submit_report(compliance_pass=false)",
                reward=obs.reward,
                done=obs.done,
                error=str(e),
            )
            return obs.result, rewards

        choice = response.choices[0]
        message = choice.message

        # Add assistant message to history
        messages.append(message.model_dump())

        # Check if the model wants to call tools
        if message.tool_calls:
            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                try:
                    fn_args = json.loads(tool_call.function.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    fn_args = {}

                action = tool_call_to_action(fn_name, fn_args)
                obs = env.step(action)

                # Format action string for logging
                args_str = ",".join(f"{k}={v}" for k, v in fn_args.items()) if fn_args else ""
                action_str = f"{fn_name}({args_str})"

                rewards.append(obs.reward)

                # Determine if there's an error in the result
                error = None
                if isinstance(obs.result, dict) and "error" in obs.result:
                    error = obs.result["error"]

                log_step(
                    step=env.step_count,
                    action=action_str,
                    reward=obs.reward,
                    done=obs.done,
                    error=error,
                )

                # Build tool response message
                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "result": obs.result,
                        "feedback": obs.system_feedback,
                        "checklist": obs.checklist,
                    }, default=str),
                }
                messages.append(tool_response)
                last_result = obs.result

                if obs.done:
                    return last_result, rewards

        elif message.content and not message.tool_calls:
            # Model responded with text instead of tool call — nudge it
            messages.append({
                "role": "user",
                "content": (
                    "Please continue your audit by calling one of the available tools. "
                    "If you are done reviewing, call submit_report with your compliance verdict."
                ),
            })

        # Check if env is done (budget exhausted)
        if env.done:
            return last_result, rewards

    # Safety: force submission if somehow we exit without done
    if not env.done:
        action = AuditAction(action_type="submit_report", compliance_pass=False)
        obs = env.step(action)
        rewards.append(obs.reward)
        log_step(
            step=env.step_count,
            action="submit_report(compliance_pass=false)",
            reward=obs.reward,
            done=obs.done,
            error=None,
        )
        last_result = obs.result

    return last_result, rewards


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    env = LocalEnvWrapper()
    all_results = []

    try:
        for task in TASKS_TO_RUN:
            task_id = task["task_id"]
            difficulty = task["difficulty"]

            start_time = time.time()
            result, rewards = run_agent_episode(env, task_id)
            elapsed = time.time() - start_time

            if result and isinstance(result, dict) and "final_score" in result:
                score = result["final_score"]
            else:
                score = 0.0

            success = score >= 0.5
            log_end(success=success, steps=len(rewards), score=score, rewards=rewards)

            all_results.append({
                "task_id": task_id,
                "difficulty": difficulty,
                "score": score,
                "time_seconds": round(elapsed, 1),
                "breakdown": result.get("breakdown", {}) if isinstance(result, dict) else {},
                "details": result.get("details", {}) if isinstance(result, dict) else {},
            })
    finally:
        env.env.close()

    # ── Executive Dashboard ─────────────────────────────────────────
    import sys

    easy_scores = [r["score"] for r in all_results if r["difficulty"] == "easy"]
    med_scores = [r["score"] for r in all_results if r["difficulty"] == "medium"]
    hard_scores = [r["score"] for r in all_results if r["difficulty"] == "hard"]

    print("\n" + "="*60, file=sys.stderr)
    print("EXECUTIVE DASHBOARD", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Model: {MODEL_NAME}", file=sys.stderr)
    print(f"Environment: {BENCHMARK}", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    print(f"Easy   ({len(easy_scores)} tasks): avg={sum(easy_scores)/len(easy_scores):.3f} min={min(easy_scores):.3f} max={max(easy_scores):.3f}", file=sys.stderr)
    print(f"Medium ({len(med_scores)} tasks): avg={sum(med_scores)/len(med_scores):.3f} min={min(med_scores):.3f} max={max(med_scores):.3f}", file=sys.stderr)
    print(f"Hard   ({len(hard_scores)} tasks): avg={sum(hard_scores)/len(hard_scores):.3f} min={min(hard_scores):.3f} max={max(hard_scores):.3f}", file=sys.stderr)
    print("-"*60, file=sys.stderr)
    all_scores = [r["score"] for r in all_results]
    total_time = sum(r["time_seconds"] for r in all_results)
    print(f"Overall: avg={sum(all_scores)/len(all_scores):.3f} | total_time={total_time:.1f}s", file=sys.stderr)
    print("="*60, file=sys.stderr)

    # Save results to JSON file
    output_file = os.getenv("OUTPUT_FILE", "baseline_results.json")
    with open(output_file, "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "benchmark": BENCHMARK,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "results": all_results,
            "summary": {
                "easy_avg": sum(easy_scores)/len(easy_scores) if easy_scores else 0,
                "medium_avg": sum(med_scores)/len(med_scores) if med_scores else 0,
                "hard_avg": sum(hard_scores)/len(hard_scores) if hard_scores else 0,
                "overall_avg": sum(all_scores)/len(all_scores) if all_scores else 0,
                "total_time": total_time,
            }
        }, f, indent=2)
    print(f"\nResults saved to {output_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
