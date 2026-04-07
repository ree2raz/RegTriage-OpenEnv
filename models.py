"""
models.py — Pydantic type definitions for the Call Center QA Environment.

Defines the OpenEnv type contract:
  - AuditAction:      What the agent sends each step (tool name + parameters)
  - AuditObservation:  What the agent receives (tool result + checklist + feedback)
  - AuditState:        Full observable episode state
  - StepResult:        Container returned by env.step()

These models enforce typed, validated input/output at every step boundary.
"""

from pydantic import BaseModel, Field
from typing import Optional, Any


class AuditAction(BaseModel):
    """What the agent can do each step.

    The agent selects one of 6 tools via action_type, passing
    tool-specific parameters as needed.

    Tools:
        get_call_metadata       — No params. Returns call context.
        get_sentiment_timeline  — No params. Returns sentiment shifts.
        read_transcript_chunk   — Requires start_turn, end_turn. Max 5 turns.
                                   Cost: 3 compute units per turn requested.
        analyze_turn            — Requires turn_index. Optional policy_hypothesis.
                                   Returns N-1, N, N+1 context + compliance rubric.
        flag_violation           — Requires violation_type, violation_severity.
                                    Optional turn_index.
        submit_report           — Requires compliance_pass. Triggers grading.
                                   Cost: 0 (always permitted).
    """
    action_type: str = Field(
        description="One of: get_call_metadata, get_sentiment_timeline, "
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


class AuditObservation(BaseModel):
    """What the agent sees after each step.

    Attributes:
        result:          Tool-specific return data (dict, list, str, etc.)
        checklist:       Current audit progress including budget gauge
        system_feedback: Human-readable guidance or error message
    """
    result: Optional[Any] = None
    checklist: dict = Field(default_factory=dict)
    system_feedback: str = ""


class AuditState(BaseModel):
    """Full observable episode state.

    Returned by env.state() to give the agent (or monitoring system)
    a complete view of the episode's current status.

    Uses compute budget instead of flat step limit:
      - total_budget: computed dynamically from transcript length
      - budget_remaining: decreases with each action
      - step_count: actual actions taken (for logging/API compatibility)
    """
    episode_id: str = ""
    difficulty: str = ""
    step_count: int = 0
    total_budget: int = 0
    budget_remaining: int = 0
    actions_taken: list[str] = Field(default_factory=list)
    flagged_violations: list[dict] = Field(default_factory=list)
    done: bool = False
    cumulative_reward: float = 0.0


class StepResult(BaseModel):
    """Returned by env.step() per OpenEnv contract.

    Bundles the observation with scalar reward, termination signal,
    and an auxiliary info dict for debugging/logging.
    """
    observation: AuditObservation
    reward: float = 0.0
    done: bool = False
    info: dict = Field(default_factory=dict)
