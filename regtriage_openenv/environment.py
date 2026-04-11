"""
environment.py — CallQAEnv: the core OpenEnv environment implementation.

Implements the OpenEnv API contract:
    reset(task_id)  → AuditObservation
    step(action)    → StepResult
    state()         → AuditState

The agent plays the role of a QA supervisor auditing call transcripts
for compliance violations using 6 strategic tools.

Tools:
    1. get_call_metadata       – Triage: call context without transcript (5 units)
    2. get_sentiment_timeline  – Hotspot detection: where sentiment shifted (5 units)
    3. read_transcript_chunk   – Strategic reading: 3 units per turn requested
    4. analyze_turn            – Contextual policy analysis: N-1, N, N+1 turns + rubric (10 units)
    5. flag_violation          – Record findings: type + severity + turn (2 units)
    6. submit_report           – Final grading: severity-weighted F1 score (0 units, always allowed)

Compute Budget:
    Actions have weighted costs reflecting real-world resource usage.
    Budget is calculated dynamically per transcript: base + (total_turns × 3).
    Episode terminates when budget is exhausted, with forced auto-submission.
"""

import json
import re
from typing import Optional, Any

from .models import AuditAction, AuditObservation, AuditState, StepResult
from .grading import grade_report
from .redact import redact_pii


# ══════════════════════════════════════════════════════════════════
# Compliance Rubric — policy definitions for analyze_turn
# ══════════════════════════════════════════════════════════════════
# In a production system, these would be loaded from a configurable
# rules engine or policy database. Hardcoded here for MVP scope,
# with a clean data structure that is obviously replaceable.

COMPLIANCE_RUBRIC = {
    "regulatory_disclosure_failure": {
        "policy": "regulatory_disclosure_failure",
        "definition": (
            "Every call MUST begin with a clear statement that the call may be "
            "recorded for quality assurance and/or training purposes. In debt "
            "collection calls, Mini-Miranda rights disclosure is also required. "
            "The disclaimer must occur within the agent's first speaking turn. "
            "Absence constitutes a regulatory violation with immediate legal liability."
        ),
        "indicators": [
            "Check the agent's first turn for phrases like 'this call may be recorded', "
            "'call is being recorded', 'recorded for quality'",
            "The disclaimer must be explicit — generic greetings without recording "
            "mention do NOT satisfy the requirement",
            "A disclaimer buried mid-call (not in the opening) is still considered missing",
            "For Collections department: also check for Mini-Miranda disclosure "
            "('this is an attempt to collect a debt')",
        ],
        "severity_guide": "Always HIGH. Recording disclaimer is a legal/regulatory requirement.",
        "common_false_positives": [
            "Agent saying 'quality service' without mentioning recording",
            "IVR/automated system providing the disclaimer before agent connects "
            "(not visible in agent transcript)",
        ],
    },
    "failed_escalation": {
        "policy": "failed_escalation",
        "definition": (
            "When a customer's sentiment reaches 'angry' (not just frustrated), the "
            "agent MUST offer to transfer the call to a supervisor or manager. An "
            "explicit customer request to speak with a supervisor must be honored "
            "immediately — deflection, delay tactics, or offering callbacks instead "
            "of live transfer all constitute failed escalation. This is a CFPB "
            "complaint trigger and regulatory investigation risk."
        ),
        "indicators": [
            "Customer uses escalation language: 'I want to speak to a supervisor', "
            "'let me talk to your manager', 'get me someone else'",
            "Customer sentiment has clearly shifted to angry: raised voice indicators, "
            "profanity, threats to leave, repeated demands",
            "Agent deflects: 'a supervisor would tell you the same thing', 'I can have "
            "someone call you back', 'let me handle this for you'",
            "Implicit escalation need: customer threatens to close account, file complaint, "
            "or contact regulatory body",
        ],
        "severity_guide": "Always HIGH. Failed escalation creates legal and reputational risk.",
        "common_false_positives": [
            "Customer is frustrated but not angry — check for clear anger indicators",
            "Agent successfully de-escalates WITH supervisor offer that customer declines",
            "Customer asks a general question about supervisor availability without "
            "requesting transfer",
        ],
    },
    "unauthorized_commitment": {
        "policy": "unauthorized_commitment",
        "definition": (
            "Agents CANNOT promise specific financial outcomes — refunds, interest rates, "
            "fee waivers, processing timelines, or policy exceptions — without documented "
            "supervisor authorization or a system-generated approval code. This creates "
            "binding verbal contracts the company may be unable to fulfill, opening the "
            "door for regulatory fines or lawsuits. Both explicit guarantees and implicit "
            "commitments that remove financial uncertainty constitute violations."
        ),
        "indicators": [
            "Explicit promises: 'I guarantee', 'I promise', 'definitely will happen', "
            "'I can guarantee that rate'",
            "Implicit commitments: 'I'll make sure that fee doesn't show up', 'don't "
            "worry about it, I'll take care of that', 'I'll just push this through'",
            "Rate/amount specifics: quoting exact percentages or dollar amounts as "
            "certain outcomes without hedging language",
            "Process bypass: 'Let me just override that', 'I'll skip the normal process'",
        ],
        "severity_guide": (
            "HIGH if financial amount > $500 or involves rate/interest commitments. "
            "MEDIUM for smaller amounts or operational commitments."
        ),
        "common_false_positives": [
            "Agent saying 'I'll submit the request for review' (submitting ≠ guaranteeing)",
            "Agent quoting published rates from an official fee schedule",
            "Agent using hedged language: 'typically', 'usually', 'you might expect'",
            "Agent confirming what has ALREADY been approved by system/supervisor",
        ],
    },
    "incorrect_hold_procedure": {
        "policy": "incorrect_hold_procedure",
        "definition": (
            "Before placing a customer on hold, the agent MUST: (1) explain why the hold "
            "is needed, (2) ask for the customer's permission, and (3) provide an estimated "
            "wait time. Going silent without notification, or extended unexplained pauses, "
            "constitute hold procedure violations. Violates TCPA regulations and drives "
            "customer abandonment."
        ),
        "indicators": [
            "Large timestamp gap between consecutive turns suggesting unannounced hold",
            "Customer saying 'hello?', 'are you still there?', indicating unexpected silence",
            "Agent resuming with 'sorry about that' after a gap without prior hold announcement",
            "Agent saying 'hold on' or 'one sec' without asking permission",
        ],
        "severity_guide": (
            "MEDIUM if customer noticed and expressed confusion. "
            "LOW if gap was brief (<30 seconds) with no customer complaint."
        ),
        "common_false_positives": [
            "Agent properly asked 'may I place you on a brief hold?' and customer agreed",
            "Brief natural pause (<5 seconds) between turns — normal conversation",
            "Customer initiated the pause ('let me grab my account number')",
        ],
    },
    "pii_exposure_risk": {
        "policy": "pii_exposure_risk",
        "definition": (
            "Agents must follow minimum-necessary PII collection principles. They must "
            "NOT request more personally identifiable information than required for the "
            "transaction. Asking for a full Social Security Number when only the last four "
            "digits are needed, reading back a full account number aloud, or requesting "
            "sensitive data already on file constitutes a PII exposure risk. This violates "
            "GDPR, CCPA, and internal data minimization policies."
        ),
        "indicators": [
            "Agent asks for full SSN: 'Can you give me your Social Security Number?' "
            "when policy requires only last four digits",
            "Agent reads back full account number aloud instead of confirming last four",
            "Agent requests information already verified by IVR or system lookup",
            "Agent asks for sensitive data (DOB, SSN, mother's maiden name) without "
            "explaining why it is needed for this specific transaction",
        ],
        "severity_guide": (
            "HIGH if full SSN or other high-sensitivity PII was requested unnecessarily. "
            "MEDIUM if agent read back full account number or requested redundant verification."
        ),
        "common_false_positives": [
            "Agent asking for last four of SSN (this is standard and correct)",
            "Agent asking for account number when no prior verification was done",
            "Agent confirming partial information: 'account ending in 4567?'",
            "Legitimate re-verification after a transfer or system timeout",
        ],
    },
    "churn_save_policy_breach": {
        "policy": "churn_save_policy_breach",
        "definition": (
            "Agents must NOT offer unauthorized discounts, credits, rate reductions, "
            "or retention incentives that are not pre-approved by the CRM retention "
            "system or a supervisor. Inventing retention offers on the spot — even if "
            "the customer is threatening to leave — constitutes a P&L leak. Retention "
            "offers must be system-driven based on Customer Lifetime Value (CLV) "
            "calculations, not agent-driven improvisation ('Hero Agent Anti-Pattern')."
        ),
        "indicators": [
            "Agent invents a discount: 'I'll give you 50% off for the next six months'",
            "Agent offers service upgrades without authorization: 'Let me bump you to "
            "our premium tier at no extra cost'",
            "Agent creates payment plans outside standard policy: 'I can spread that "
            "over 12 months interest-free' without system approval",
            "Agent offers credits to prevent churn: 'How about I credit $100 to your "
            "account to make up for the inconvenience?'",
            "Key difference from unauthorized_commitment: churn_save is about giving "
            "away the company's money to retain a customer. unauthorized_commitment is "
            "about promising a financial outcome the company may not be able to deliver.",
        ],
        "severity_guide": (
            "HIGH if the unauthorized retention offer exceeds $200 or involves "
            "multi-month commitments. MEDIUM for smaller one-time credits or minor "
            "service adjustments."
        ),
        "common_false_positives": [
            "Agent offering a retention deal that IS in the CRM-approved offers list",
            "Agent saying 'I can check if any promotions are available for your account'",
            "Agent executing a supervisor-approved exception with documented approval",
            "Standard courtesy credits within the agent's documented authority level",
        ],
    },
}


# ══════════════════════════════════════════════════════════════════
# Action Costs — weighted by real-world resource usage
# ══════════════════════════════════════════════════════════════════

ACTION_COSTS = {
    "get_call_metadata": 5,       # Quick DB lookup, small payload
    "get_sentiment_timeline": 5,  # Pre-computed data
    "read_transcript_chunk": 3,   # Per turn requested (dynamic)
    "analyze_turn": 10,           # Contextual analysis + rubric
    "flag_violation": 2,          # Simple state recording
    "submit_report": 0,           # Terminal, always allowed
}

# Minimum cost of any non-terminal, non-flag action
MIN_MEANINGFUL_COST = 5


class CallQAEnv:
    """
    RegTriage — Financial Services Compliance Auditing Environment.

    The agent audits call transcripts by strategically using tools to:
    1. Triage via metadata and sentiment hotspots
    2. Drill into specific transcript sections
    3. Flag violations with type, turn, and severity
    4. Submit a final compliance report (triggers grading)

    Uses a compute budget with weighted action costs instead of a flat
    step limit. Budget scales dynamically with transcript length.

    Scoring uses severity-weighted F1 with auto-fail for missed critical
    violations. See grading.py for the complete scoring specification.
    """

    CHUNK_MAX_TURNS = 5  # max turns per read_transcript_chunk call

    def __init__(self, transcript_path: str = None):
        if transcript_path is None:
            # Try to find transcripts.json relative to package
            import os
            transcript_path = os.path.join(os.path.dirname(__file__), "..", "transcripts.json")
            if not os.path.exists(transcript_path):
                transcript_path = "transcripts.json"
        with open(transcript_path) as f:
            all_transcripts = json.load(f)
        self.transcripts = {t["id"]: t for t in all_transcripts}
        self._reset_state()

    def _reset_state(self):
        """Clear all episode state."""
        self.current = None
        self.step_count = 0
        self.total_budget = 0
        self.budget_remaining = 0
        self.flagged_violations: list[dict] = []
        self.actions_taken: list[str] = []
        self.done = False
        self.cumulative_reward = 0.0
        self._entity_names: dict[str, str] = {}  # PII redaction map

    def _extract_entity_names(self, transcript: dict) -> dict[str, str]:
        """Extract speaker names from transcript for PII redaction.

        Scans the opening turns for name introductions and builds a mapping
        from name → redaction token. Only matches proper nouns (uppercase-initial).

        In production, this would be a dedicated NER pipeline. For MVP,
        regex extraction from opening turns is sufficient.
        """
        entities = {}
        # Only match names that start with uppercase (proper nouns)
        name_patterns = [
            # "My name is John Chen" / "I'm Sarah" / "I am Brian"
            re.compile(r'(?:[Mm]y name is|I\'m|I am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'),
            # "This is Maria Lopez" — only at sentence start or after greeting
            re.compile(r'(?:^|[.!?]\s+|,\s+)[Tt]his is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'),
        ]

        # Common false positives to skip
        skip_names = {"Acme", "Financial", "Services", "Thank", "Hello", "Sure"}

        # Only scan first 8 turns — names are introduced early
        for turn in transcript["turns"][:8]:
            for pattern in name_patterns:
                matches = pattern.findall(turn["text"])
                for name in matches:
                    name = name.strip()
                    # Must be at least 2 chars and not a known false positive
                    if len(name) < 2 or name in skip_names:
                        continue
                    # Must start with uppercase (double-check)
                    if not name[0].isupper():
                        continue
                    token = "[AGENT_NAME]" if turn["speaker"] == "agent" else "[CUSTOMER_NAME]"
                    entities[name] = token
                    # Also add first name alone for later turn references
                    first_name = name.split()[0]
                    if first_name not in skip_names and first_name not in entities:
                        entities[first_name] = token

        return entities

    def _calculate_budget(self, transcript: dict) -> int:
        """Calculate compute budget based on transcript complexity.

        Formula: base + (total_turns × 3)
        - base=50 covers minimum viable audit (triage + submit)
        - per-turn allowance gives budget proportional to data size

        Examples:
            Easy  (10 turns): 50 + 30  = 80 units
            Medium (20 turns): 50 + 60  = 110 units
            Hard  (35 turns): 50 + 105 = 155 units
        """
        total_turns = len(transcript["turns"])
        return 50 + (total_turns * 3)

    def _get_action_cost(self, action: AuditAction) -> int:
        """Get the compute cost for an action.

        read_transcript_chunk is priced per turn requested (3 units/turn),
        not a flat rate. Rewards precision reading.
        All other actions have fixed costs from ACTION_COSTS.
        """
        if action.action_type == "read_transcript_chunk":
            if action.start_turn is not None and action.end_turn is not None:
                turns_requested = action.end_turn - action.start_turn + 1
                return ACTION_COSTS["read_transcript_chunk"] * turns_requested
            return ACTION_COSTS["read_transcript_chunk"] * self.CHUNK_MAX_TURNS  # assume max if invalid

        return ACTION_COSTS.get(action.action_type, MIN_MEANINGFUL_COST)

    # ── OpenEnv API ───────────────────────────────────────────

    def reset(self, task_id: Optional[str] = None) -> AuditObservation:
        """Load a specific transcript and reset all state.

        Args:
            task_id: Transcript ID (e.g., "call_001"). Defaults to first available.

        Returns:
            Initial observation with call summary and empty checklist.
        """
        if task_id is None:
            task_id = list(self.transcripts.keys())[0]
        if task_id not in self.transcripts:
            raise ValueError(
                f"Unknown task_id: {task_id}. "
                f"Available: {list(self.transcripts.keys())}"
            )
        self._reset_state()
        self.current = self.transcripts[task_id]
        self.total_budget = self._calculate_budget(self.current)
        self.budget_remaining = self.total_budget
        self._entity_names = self._extract_entity_names(self.current)

        return AuditObservation(
            result=(
                f"Loaded call {task_id} | "
                f"{len(self.current['turns'])} turns | "
                f"Department: {self.current['metadata']['department']} | "
                f"Compute budget: {self.total_budget} units"
            ),
            checklist=self._build_checklist(),
            system_feedback="Environment reset. Use get_call_metadata to begin your audit."
        )

    def step(self, action: AuditAction) -> StepResult:
        """Process one agent action, return StepResult.

        Dispatches to the appropriate tool, applies budget enforcement,
        and accumulates rewards.

        Budget rules:
        - submit_report always costs 0 and is always allowed
        - If budget would go negative, auto-submit with penalty
        - Budget warning when only flag/submit remain affordable
        """
        if self.done:
            return StepResult(
                observation=AuditObservation(system_feedback="Episode already finished."),
                reward=0.0, done=True, info={}
            )
        if self.current is None:
            return StepResult(
                observation=AuditObservation(system_feedback="Call reset() with a task_id first."),
                reward=0.0, done=True, info={}
            )

        # ── Budget check before execution ─────────────────
        action_cost = self._get_action_cost(action)

        # submit_report is always free and always allowed
        if action.action_type != "submit_report" and action_cost > self.budget_remaining:
            # Cannot afford this action — auto-submit with penalty
            self.done = True
            self.step_count += 1
            self.actions_taken.append("auto_submit_budget_exhausted")

            result, reward = self._tool_submit_report(False)
            reward -= 0.10  # budget exhaustion penalty
            self.cumulative_reward += reward

            obs = AuditObservation(
                result=result,
                checklist=self._build_checklist(),
                system_feedback=(
                    f"Budget exhausted ({self.budget_remaining} remaining, "
                    f"action costs {action_cost}). Auto-submitting with penalty."
                )
            )
            return StepResult(observation=obs, reward=reward, done=True,
                              info={"step": self.step_count, "auto_submit": True,
                                    "cumulative_reward": self.cumulative_reward})

        # ── Deduct budget and execute ─────────────────────
        self.budget_remaining -= action_cost
        self.step_count += 1
        self.actions_taken.append(action.action_type)

        # ── Tool dispatch ─────────────────────────────────
        result, reward, feedback = self._dispatch(action)

        # ── Budget warning ────────────────────────────────
        if not self.done and self.budget_remaining < MIN_MEANINGFUL_COST:
            if self.budget_remaining >= ACTION_COSTS["flag_violation"]:
                feedback += " ⚠ Budget critically low. Only flagging and submission remain affordable."
            else:
                feedback += " ⚠ Budget exhausted. Submit your report now."

        self.cumulative_reward += reward

        obs = AuditObservation(
            result=result,
            checklist=self._build_checklist(),
            system_feedback=feedback
        )

        return StepResult(
            observation=obs,
            reward=reward,
            done=self.done,
            info={"step": self.step_count, "budget_used": action_cost,
                  "cumulative_reward": self.cumulative_reward}
        )

    def state(self) -> AuditState:
        """Return full observable episode state."""
        return AuditState(
            episode_id=self.current["id"] if self.current else "",
            difficulty=self.current["difficulty"] if self.current else "",
            step_count=self.step_count,
            total_budget=self.total_budget,
            budget_remaining=self.budget_remaining,
            actions_taken=self.actions_taken.copy(),
            flagged_violations=[v.copy() for v in self.flagged_violations],
            done=self.done,
            cumulative_reward=self.cumulative_reward,
        )

    # ── Tool Dispatch ─────────────────────────────────────────

    def _dispatch(self, action: AuditAction) -> tuple[Any, float, str]:
        """Route action to the correct tool. Returns (result, reward, feedback)."""
        tool = action.action_type

        if tool == "get_call_metadata":
            return self._tool_get_call_metadata(), 0.05, "Metadata returned. Review before reading transcript."

        elif tool == "get_sentiment_timeline":
            return self._tool_get_sentiment_timeline(), 0.05, "Sentiment timeline returned. Use hotspots to target your investigation."

        elif tool == "read_transcript_chunk":
            result = self._tool_read_transcript_chunk(action.start_turn, action.end_turn)
            if "error" in result:
                return result, -0.02, f"Error: {result['error']}"
            return result, 0.02, f"Transcript chunk returned (turns {action.start_turn}-{action.end_turn})."

        elif tool == "analyze_turn":
            result = self._tool_analyze_turn(action.turn_index, action.policy_hypothesis)
            if "error" in result:
                return result, -0.02, f"Error: {result['error']}"
            return result, 0.02, f"Contextual analysis of turn {action.turn_index} complete."

        elif tool == "flag_violation":
            result = self._tool_flag_violation(action)
            if "error" in result:
                return result, -0.02, f"Error: {result['error']}"
            return result, 0.0, "Violation flagged. Score determined at submission."

        elif tool == "submit_report":
            result, reward = self._tool_submit_report(action.compliance_pass)
            self.done = True
            return result, reward, "Report submitted. Episode complete."

        else:
            return None, -0.05, f"Unknown action_type: {tool}"

    # ── Tool Implementations ──────────────────────────────────

    def _tool_get_call_metadata(self) -> dict:
        """Return call metadata WITHOUT transcript content.
        Agent must decide what to investigate based on this."""
        meta = self.current["metadata"]
        return {
            "call_id": self.current["id"],
            "difficulty": self.current["difficulty"],
            "department": meta["department"],
            "call_reason": meta["call_reason"],
            "call_duration_seconds": meta["call_duration_seconds"],
            "call_summary": meta["call_summary"],
            "total_turns": len(self.current["turns"]),
        }

    def _tool_get_sentiment_timeline(self) -> list[dict]:
        """Return sentiment shift markers from ground truth.
        Agent uses these as hotspot indicators for targeted investigation."""
        gt = self.current["ground_truth"]
        return gt.get("customer_sentiment_shifts", [])

    def _tool_read_transcript_chunk(self, start: Optional[int], end: Optional[int]) -> dict:
        """Return a chunk of the transcript (max CHUNK_MAX_TURNS turns).
        Forces strategic reading — agent cannot dump the whole thing.
        Cost scales with turns requested: 3 units per turn."""
        turns = self.current["turns"]

        if start is None or end is None:
            return {"error": "Provide start_turn and end_turn parameters."}
        if start < 0 or end >= len(turns) or start > end:
            return {"error": f"Invalid range [{start}, {end}]. Valid: [0, {len(turns) - 1}]."}
        if (end - start + 1) > self.CHUNK_MAX_TURNS:
            return {
                "error": f"Chunk too large ({end - start + 1} turns). "
                         f"Maximum {self.CHUNK_MAX_TURNS} turns per read."
            }

        chunk = [
            {
                "turn_index": t["turn_index"],
                "speaker": t["speaker"],
                "text": redact_pii(t["text"], self._entity_names),
                "timestamp_start": t["timestamp_start"],
                "timestamp_end": t["timestamp_end"],
            }
            for t in turns[start:end + 1]
        ]
        return {"turns": chunk, "range": [start, end], "total_turns": len(turns)}

    def _tool_analyze_turn(self, turn_index: Optional[int],
                           policy_hypothesis: Optional[str] = None) -> dict:
        """Contextual policy analysis of a single turn.

        Unlike read_transcript_chunk (which returns raw text), this tool
        provides analytical scaffolding that mirrors how a real QA supervisor
        reviews utterances:

        1. Context window: Returns the target turn plus N-1 and N+1 turns
           for conversational state (was hold permission asked? did agent
           acknowledge the wait?)

        2. Temporal signals: Silence gaps indicating potential hold violations

        3. Position awareness: Whether the turn is in the opening, mid-call,
           or closing segment

        4. Policy rubric (if policy_hypothesis provided): The full compliance
           definition for the hypothesized violation, so the agent can
           cross-reference the utterance against the regulatory standard.
        """
        if turn_index is None:
            return {"error": "Provide turn_index parameter."}
        turns = self.current["turns"]
        if turn_index < 0 or turn_index >= len(turns):
            return {"error": f"turn_index {turn_index} out of range [0, {len(turns) - 1}]."}

        turn = turns[turn_index]
        total = len(turns)

        # ── Build context window (N-1, N, N+1) ───────────
        context_before = None
        context_after = None

        if turn_index > 0:
            prev = turns[turn_index - 1]
            context_before = {
                "turn_index": prev["turn_index"],
                "speaker": prev["speaker"],
                "text": redact_pii(prev["text"], self._entity_names),
                "timestamp_end": prev["timestamp_end"],
            }

        if turn_index < total - 1:
            nxt = turns[turn_index + 1]
            context_after = {
                "turn_index": nxt["turn_index"],
                "speaker": nxt["speaker"],
                "text": redact_pii(nxt["text"], self._entity_names),
                "timestamp_start": nxt["timestamp_start"],
            }

        # ── Temporal signals ──────────────────────────────
        silence_before = 0.0
        if turn_index > 0:
            prev_end = turns[turn_index - 1]["timestamp_end"]
            silence_before = round(turn["timestamp_start"] - prev_end, 1)

        # ── Position awareness ────────────────────────────
        if turn_index <= 2:
            position = "opening"
        elif turn_index >= total - 3:
            position = "closing"
        else:
            position = "mid-call"

        result = {
            "turn_index": turn_index,
            "speaker": turn["speaker"],
            "text": redact_pii(turn["text"], self._entity_names),
            "timestamp_start": turn["timestamp_start"],
            "timestamp_end": turn["timestamp_end"],
            "context_before": context_before,
            "context_after": context_after,
            "silence_gap_before_seconds": silence_before,
            "speaker_turn_position": position,
        }

        # ── Policy rubric lookup ──────────────────────────
        if policy_hypothesis:
            rubric = COMPLIANCE_RUBRIC.get(policy_hypothesis)
            if rubric:
                result["compliance_rubric"] = rubric
            else:
                result["compliance_rubric_error"] = (
                    f"Unknown policy: '{policy_hypothesis}'. "
                    f"Valid policies: {list(COMPLIANCE_RUBRIC.keys())}"
                )

        return result

    def _tool_flag_violation(self, action: AuditAction) -> dict:
        """Record a violation finding. No score feedback until submission."""
        valid_types = set(COMPLIANCE_RUBRIC.keys())
        if not action.violation_type:
            return {"error": "Provide violation_type parameter."}
        if action.violation_type not in valid_types:
            return {"error": f"Invalid violation_type '{action.violation_type}'. Valid: {valid_types}"}
        if action.violation_severity not in ("high", "medium", "low", None):
            return {"error": f"Invalid violation_severity. Must be high, medium, or low."}

        violation = {
            "type": action.violation_type,
            "turn_index": action.turn_index,
            "severity": action.violation_severity or "low",
        }
        self.flagged_violations.append(violation)
        return {"status": "recorded", "violation": violation, "total_flagged": len(self.flagged_violations)}

    def _tool_submit_report(self, compliance_pass: Optional[bool]) -> tuple[dict, float]:
        """Grade the agent's audit against ground truth.

        Delegates to grading.grade_report() for the actual scoring.
        See grading.py for the complete scoring specification.
        """
        gt = self.current["ground_truth"]
        agent_compliance = compliance_pass if compliance_pass is not None else True

        return grade_report(
            gt_violations=gt["violations"],
            gt_compliance=gt["overall_compliance_pass"],
            flagged_violations=self.flagged_violations,
            agent_compliance=agent_compliance,
            budget_remaining=self.budget_remaining,
            total_budget=self.total_budget,
        )

    # ── Helpers ───────────────────────────────────────────────

    def _build_checklist(self) -> dict:
        """Build the current audit progress checklist with budget gauge."""
        pct = round(100 * self.budget_remaining / self.total_budget) if self.total_budget > 0 else 0
        return {
            "metadata_reviewed": "get_call_metadata" in self.actions_taken,
            "sentiment_checked": "get_sentiment_timeline" in self.actions_taken,
            "transcript_chunks_read": self.actions_taken.count("read_transcript_chunk"),
            "turns_analyzed": self.actions_taken.count("analyze_turn"),
            "violations_flagged": len(self.flagged_violations),
            "report_submitted": self.done and ("submit_report" in self.actions_taken
                                               or "auto_submit_budget_exhausted" in self.actions_taken),
            "budget_remaining_pct": pct,
            "budget_remaining_units": self.budget_remaining,
            "total_budget": self.total_budget,
        }

    def get_available_tasks(self) -> list[dict]:
        """List all available tasks with their difficulty levels."""
        return [
            {"task_id": tid, "difficulty": t["difficulty"]}
            for tid, t in self.transcripts.items()
        ]
    
    def close(self):
        """Close the environment. Required for OpenEnv compatibility."""
        # No resources to clean up in this implementation
        pass
    
    def get_metadata(self) -> dict:
        """Get environment metadata. Required for OpenEnv compatibility."""
        return {
            "name": "regtriage",
            "description": "Financial services regulatory compliance auditing environment",
            "version": "2.0.1",
        }
