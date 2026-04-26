"""Pydantic models for the RegTriage OpenEnv environment.

These dataclasses define the wire format that flows between the OpenEnv
server and any EnvClient (local, Docker, or HF Spaces).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# OpenEnv base types - imported directly from openenv-core (required)
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class AuditAction(Action):
    """What the agent can do each step.

    The agent selects one of 7 tools via action_type, passing
    tool-specific parameters as needed.

    Tools:
        get_call_metadata       — No params. Returns call context.
        get_sentiment_timeline  — No params. Returns sentiment shifts.
        get_transcript_length   — No params. Returns turn count and valid range. Cost: 1.
        read_transcript_chunk   — Requires start_turn, end_turn. Max 5 turns.
                                   Cost: 3 compute units per turn requested.
        analyze_turn            — Requires turn_index. Optional policy_hypothesis.
                                   Returns N-1, N, N+1 context + compliance rubric.
        flag_violation          — Requires violation_type, violation_severity.
                                    Optional turn_index.
        submit_report           — Requires compliance_pass. Triggers grading.
                                   Cost: 0 (always permitted).
    """
    action_type: str = Field(
        default="get_call_metadata",
        description="One of: get_call_metadata, get_sentiment_timeline, get_transcript_length, "
                    "read_transcript_chunk, analyze_turn, flag_violation, submit_report"
    )
    # Tool-specific parameters (optional depending on action_type)
    turn_index: Optional[int] = Field(default=None, description="Target turn for analyze_turn or flag_violation")
    start_turn: Optional[int] = Field(default=None, description="Start of range for read_transcript_chunk")
    end_turn: Optional[int] = Field(default=None, description="End of range for read_transcript_chunk")
    violation_type: Optional[str] = Field(default=None, description="Violation category for flag_violation")
    violation_severity: Optional[str] = Field(default=None, description="high, medium, or low")
    compliance_pass: Optional[bool] = Field(default=None, description="Agent's overall compliance verdict for submit_report")
    policy_hypothesis: Optional[str] = Field(
        default=None,
        description="Policy to check against for analyze_turn (e.g., 'unauthorized_commitment'). "
                    "If provided, the compliance rubric definition for that policy is included in the response."
    )


# ---------------------------------------------------------------------------
# Checklist (auxiliary model)
# ---------------------------------------------------------------------------

class AuditChecklist(BaseModel):
    """Tracks the agent's audit progress through the episode.

    Shown in every observation so the agent can plan next steps.
    Includes compute budget as a percentage fuel gauge.
    """
    metadata_reviewed: bool = False
    sentiment_checked: bool = False
    transcript_chunks_read: int = 0
    turns_analyzed: int = 0
    violations_flagged: int = 0
    report_submitted: bool = False
    budget_remaining_pct: int = 100
    budget_remaining_units: int = 0
    total_budget: int = 0


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class AuditObservation(Observation):
    """What the agent sees after each step.

    Attributes:
        result:          Tool-specific return data (dict, list, str, etc.)
        checklist:       Current audit progress including budget gauge
        system_feedback: Human-readable guidance or error message
    """
    result: Optional[Any] = Field(default=None, description="Tool-specific return data")
    checklist: Dict[str, Any] = Field(default_factory=dict, description="Current audit progress")
    system_feedback: str = Field(default="", description="Human-readable guidance or error message")


# ---------------------------------------------------------------------------
# State (internal server state)
# ---------------------------------------------------------------------------

class AuditState(State):
    """Full observable episode state.

    Returned by env.state() to give the agent (or monitoring system)
    a complete view of the episode's current status.

    Uses compute budget instead of flat step limit:
      - total_budget: computed dynamically from transcript length
      - budget_remaining: decreases with each action
      - step_count: actual actions taken (for logging/API compatibility)
    """
    episode_id: str = Field(default="", description="Unique episode identifier")
    difficulty: str = Field(default="", description="Task difficulty level")
    step_count: int = Field(default=0, description="Number of steps taken")
    total_budget: int = Field(default=0, description="Total compute budget available")
    budget_remaining: int = Field(default=0, description="Remaining compute budget")
    actions_taken: List[str] = Field(default_factory=list, description="History of actions")
    flagged_violations: List[Dict[str, Any]] = Field(default_factory=list, description="Violations flagged")
    done: bool = Field(default=False, description="Whether episode is complete")
    cumulative_reward: float = Field(default=0.0, description="Cumulative reward so far")


# ---------------------------------------------------------------------------
# Transcript Models (for structured data)
# ---------------------------------------------------------------------------

class Turn(BaseModel):
    """A single turn in a call transcript."""
    turn_index: int
    speaker: str
    text: str
    timestamp_start: float
    timestamp_end: float


class Violation(BaseModel):
    """A compliance violation in ground truth."""
    type: str
    description: str
    turn_index: Optional[int]
    severity: str


class GroundTruth(BaseModel):
    """Ground truth data for a transcript."""
    disclaimer_present: bool
    disclaimer_turn_index: Optional[int]
    escalation_required: bool
    escalation_performed: bool
    customer_sentiment_shifts: List[Dict[str, Any]]
    violations: List[Violation]
    overall_compliance_pass: bool


class Transcript(BaseModel):
    """A complete call transcript with metadata."""
    id: str
    difficulty: str
    metadata: Dict[str, Any]
    turns: List[Turn]
    ground_truth: GroundTruth
