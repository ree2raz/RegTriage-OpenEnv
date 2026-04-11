"""
inference.py — Baseline inference script for RegTriage OpenEnv environment.

Runs an LLM agent against the environment using the OpenAI-compatible API.
The agent uses function calling (tool use) to strategically audit call transcripts.

Two modes:
1. Local mode (default): Uses local CallQAEnv directly
2. Client mode (--use-client): Connects via RegTriageEnv client to running server

Required environment variables (per hackathon spec):
  API_BASE_URL    — API endpoint (default provided)
  MODEL_NAME      — Model identifier (default provided)
  HF_TOKEN        — API key (no default — required)

STDOUT FORMAT (per hackathon spec):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...,rn>

Usage:
  # Local mode (recommended)
  uv run python inference.py
  
  # Client mode (connect to running server)
  uv run python inference.py --use-client --env-url http://localhost:7860
"""

import argparse
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
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
BENCHMARK = "regtriage"

if not HF_TOKEN:
    raise RuntimeError(
        "No API key found. Set HF_TOKEN in environment or .env file."
    )

client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

# Run ALL 12 tasks — demonstrates confidence and exercises full task suite
TASKS_TO_RUN = [
    {"task_id": "call_001", "difficulty": "easy"},
    {"task_id": "call_002", "difficulty": "easy"},
    {"task_id": "call_003", "difficulty": "easy"},
    {"task_id": "call_004", "difficulty": "easy"},
    {"task_id": "call_005", "difficulty": "medium"},
    {"task_id": "call_006", "difficulty": "medium"},
    {"task_id": "call_007", "difficulty": "medium"},
    {"task_id": "call_008", "difficulty": "medium"},
    {"task_id": "call_009", "difficulty": "hard"},
    {"task_id": "call_010", "difficulty": "hard"},
    {"task_id": "call_011", "difficulty": "hard"},
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
        obs = self.env.reset(task_id)
        self.step_count = 0
        return obs
    
    def step(self, action: AuditAction):
        result = self.env.step(action)
        self.step_count = self.env.step_count
        return result
    
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
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
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
- read_transcript_chunk: 3 units PER TURN requested (reading 5 turns = 15 units)
- analyze_turn: 10 units (includes context window + optional policy rubric)
- flag_violation: 2 units (cheap — flag freely)
- submit_report: 0 units (always allowed)

Your budget remaining is shown as a percentage after each step. Be efficient:
use metadata and sentiment to triage, then read only the sections that matter.

## Your Audit Workflow
1. **Start with metadata** — call get_call_metadata() to understand the call context.
2. **Check sentiment** — call get_sentiment_timeline() to identify emotional hotspots.
3. **Read the opening** — read the first 2-3 turns to check for the mandatory recording disclaimer.
4. **Read around hotspots** — read turns around any sentiment shifts to check escalation handling.
5. **Check PII handling** — look for moments where identity verification occurs.
6. **Analyze suspicious turns** — use analyze_turn() with a policy_hypothesis to get the
   compliance rubric and cross-reference the utterance against the standard.
7. **Flag ALL violations** — for each violation found, flag it with type, turn index, and severity.
8. **Submit report** — when done, submit your final compliance verdict.

## Compliance Rules — Taxonomy of Risk (CHECK ALL)
### Legal Liability Violations
- **Regulatory Disclosure Failure (ALWAYS high severity)**: Every call MUST begin with a statement
  that the call may be recorded for quality assurance and/or training purposes. In debt
  collection calls, Mini-Miranda rights disclosure is also required.
  The disclaimer must occur within the agent's first speaking turn.
  Absence constitutes a regulatory violation with immediate legal liability.
- **Failed Escalation (ALWAYS high severity)**: When customer sentiment reaches "angry", agent
  MUST offer to transfer to a supervisor. Deflection = flag as failed_escalation.
- **PII Exposure Risk (high-medium severity)**: Agents must follow data minimization. Asking for
  FULL SSN when only last 4 needed, or reading back full account numbers aloud = flag as
  pii_exposure_risk.

### Revenue Leakage Violations
- **Unauthorized Commitment (high-medium severity)**: Agents CANNOT promise specific refunds,
  interest rates, outcomes without supervisor approval. Creates binding verbal contracts.
  Flag as unauthorized_commitment.
- **Churn Save Policy Breach (high-medium severity)**: Agents CANNOT invent discounts, credits,
  or retention offers not pre-approved by CRM system. Even if customer is threatening to leave,
  the agent cannot improvise financial concessions. Flag as churn_save_policy_breach.

### Operational Violations
- **Incorrect Hold Procedure (medium-low severity)**: Before placing customer on hold, agent MUST
  ask permission. Going silent without warning = flag as incorrect_hold_procedure.

## CRITICAL RULES
- Missing a high-severity violation severely reduces the score. Be thorough.
- A happy customer does NOT mean a compliant call. Watch for the 'Hero Agent' who breaks rules
  to make the customer happy.
- Flag ALL violations — not just the obvious ones.
- Set compliance_pass to false if ANY violations exist, true ONLY if the call is fully clean.
- Some calls may be fully compliant — do not flag violations that don't exist.
- Use analyze_turn with policy_hypothesis to get the compliance rubric when unsure.
- Be efficient with your budget — precision, not brute force.
"""

# ══════════════════════════════════════════════════════════════════
# Agent Loop
# ══════════════════════════════════════════════════════════════════


def tool_call_to_action(tool_name: str, arguments: dict) -> AuditAction:
    """Convert an OpenAI tool call into an AuditAction."""
    return AuditAction(
        action_type=tool_name,
        turn_index=arguments.get("turn_index"),
        start_turn=arguments.get("start_turn"),
        end_turn=arguments.get("end_turn"),
        violation_type=arguments.get("violation_type"),
        violation_severity=arguments.get("violation_severity"),
        compliance_pass=arguments.get("compliance_pass"),
        policy_hypothesis=arguments.get("policy_hypothesis"),
    )


def run_agent_episode(env: LocalEnvWrapper, task_id: str) -> tuple[dict | None, list[float]]:
    """
    Run one complete episode: reset env, let LLM make tool calls until done.
    Returns (final_result_dict, list_of_rewards).
    """
    obs = env.reset(task_id)
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Audit this call. {obs.system_feedback}\n"
                f"Call info: {obs.result}\n"
                "Begin your audit by calling get_call_metadata()."
            ),
        },
    ]

    last_result = None
    rewards: list[float] = []
    max_iterations = 30  # safety margin (budget handles episode length)

    for iteration in range(max_iterations):
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
            step_result = env.step(action)
            rewards.append(step_result.reward)
            log_step(
                step=env.step_count,
                action="submit_report(compliance_pass=false)",
                reward=step_result.reward,
                done=step_result.done,
                error=str(e),
            )
            return step_result.observation.result, rewards

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
                step_result = env.step(action)

                # Format action string for logging
                args_str = ",".join(f"{k}={v}" for k, v in fn_args.items()) if fn_args else ""
                action_str = f"{fn_name}({args_str})"

                rewards.append(step_result.reward)

                # Determine if there's an error in the result
                error = None
                if isinstance(step_result.observation.result, dict) and "error" in step_result.observation.result:
                    error = step_result.observation.result["error"]

                log_step(
                    step=env.step_count,
                    action=action_str,
                    reward=step_result.reward,
                    done=step_result.done,
                    error=error,
                )

                # Build tool response message
                tool_response = {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "result": step_result.observation.result,
                        "feedback": step_result.observation.system_feedback,
                        "checklist": step_result.observation.checklist,
                    }, default=str),
                }
                messages.append(tool_response)
                last_result = step_result.observation.result

                if step_result.done:
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
        step_result = env.step(action)
        rewards.append(step_result.reward)
        log_step(
            step=env.step_count,
            action="submit_report(compliance_pass=false)",
            reward=step_result.reward,
            done=step_result.done,
            error=None,
        )
        last_result = step_result.observation.result

    return last_result, rewards


# ══════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="RegTriage Baseline Inference")
    parser.add_argument(
        "--use-client",
        action="store_true",
        help="Use RegTriageEnv client instead of local environment",
    )
    parser.add_argument(
        "--env-url",
        type=str,
        default="http://localhost:7860",
        help="Environment server URL (when using --use-client)",
    )
    
    args = parser.parse_args()
    
    # Initialize environment
    if args.use_client:
        from regtriage_openenv.client import RegTriageEnv
        print(f"Connecting to environment at {args.env_url}...")
        env_client = RegTriageEnv(base_url=args.env_url)
        env_client.__enter__()
        # Note: Client interface differs slightly, this is a simplified wrapper
        env = LocalEnvWrapper()  # Fallback to local for now
        print("Note: Using local environment (client mode implementation pending)")
    else:
        env = LocalEnvWrapper()
    
    all_results = []

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

    # Cleanup
    if args.use_client:
        env_client.__exit__(None, None, None)


if __name__ == "__main__":
    main()
