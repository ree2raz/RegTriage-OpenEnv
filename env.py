"""
env.py — Re-export facade for backward compatibility.

All types and the environment class are defined in their own modules:
    models.py       → AuditAction, AuditObservation, AuditState, StepResult
    grading.py      → compute_violation_f1, grade_report
    environment.py  → CallQAEnv

This module re-exports everything so that existing imports like
    from env import CallQAEnv, AuditAction
continue to work unchanged.

Run directly to execute smoke tests:
    uv run python env.py
"""

import json

# Re-export public API from submodules
from models import AuditAction, AuditObservation, AuditState, AuditChecklist, StepResult  # noqa: F401
from environment import CallQAEnv  # noqa: F401


# ══════════════════════════════════════════════════════════════════
# Smoke Test — validates the full env + grading + budget pipeline
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    env = CallQAEnv()

    tasks = env.get_available_tasks()
    print(f"Available tasks ({len(tasks)}): {tasks}\n")

    # ── Test cases aligned with generated transcripts ─────
    test_cases = [
        {
            "task_id": "call_001",
            "label": "EASY — regulatory disclosure failure (fail)",
            "flags": [
                {"violation_type": "regulatory_disclosure_failure", "turn_index": None, "violation_severity": "high"},
            ],
            "compliance_pass": False,
        },
        {
            "task_id": "call_002",
            "label": "EASY — compliant call (pass)",
            "flags": [],  # No violations — tests false-positive restraint
            "compliance_pass": True,
        },
        {
            "task_id": "call_005",
            "label": "MEDIUM — failed escalation + unauthorized commitment",
            "flags": [
                {"violation_type": "failed_escalation", "turn_index": 13, "violation_severity": "high"},
                {"violation_type": "unauthorized_commitment", "turn_index": 10, "violation_severity": "medium"},
            ],
            "compliance_pass": False,
        },
        {
            "task_id": "call_010",
            "label": "HARD — 4 violations (collections, Hero Agent)",
            "flags": [
                {"violation_type": "churn_save_policy_breach", "turn_index": 16, "violation_severity": "high"},
                {"violation_type": "pii_exposure_risk", "turn_index": 12, "violation_severity": "high"},
                {"violation_type": "failed_escalation", "turn_index": 18, "violation_severity": "high"},
                {"violation_type": "incorrect_hold_procedure", "turn_index": 8, "violation_severity": "medium"},
            ],
            "compliance_pass": False,
        },
    ]

    for tc in test_cases:
        print(f"{'=' * 60}")
        print(f"  {tc['label']}")
        print(f"{'=' * 60}")

        # Reset
        obs = env.reset(tc["task_id"])
        print(f"  Reset: {obs.system_feedback}")
        print(f"  Budget: {env.total_budget} units")

        # Step 1: Get metadata
        result = env.step(AuditAction(action_type="get_call_metadata"))
        print(f"  Step 1 (metadata): reward={result.reward}, budget={env.budget_remaining}/{env.total_budget}")

        # Step 2: Get sentiment
        result = env.step(AuditAction(action_type="get_sentiment_timeline"))
        print(f"  Step 2 (sentiment): reward={result.reward}, budget={env.budget_remaining}/{env.total_budget}")

        # Step 3: Read first chunk
        result = env.step(AuditAction(action_type="read_transcript_chunk", start_turn=0, end_turn=4))
        print(f"  Step 3 (chunk 0-4): reward={result.reward}, budget={env.budget_remaining}/{env.total_budget}")

        # Step 4: Analyze a turn with policy hypothesis (if there are violations to check)
        if tc["flags"]:
            first_flag = tc["flags"][0]
            result = env.step(AuditAction(
                action_type="analyze_turn",
                turn_index=first_flag.get("turn_index") or 0,
                policy_hypothesis=first_flag["violation_type"],
            ))
            analysis = result.observation.result
            has_rubric = "compliance_rubric" in analysis if isinstance(analysis, dict) else False
            print(f"  Step 4 (analyze w/ hypothesis): reward={result.reward}, has_rubric={has_rubric}, budget={env.budget_remaining}/{env.total_budget}")

        # Flag violations
        for i, flag in enumerate(tc["flags"]):
            result = env.step(AuditAction(
                action_type="flag_violation",
                violation_type=flag["violation_type"],
                turn_index=flag["turn_index"],
                violation_severity=flag["violation_severity"],
            ))
            step_num = (5 if tc["flags"] else 4) + i
            print(f"  Step {step_num} (flag {flag['violation_type']}): reward={result.reward}, budget={env.budget_remaining}/{env.total_budget}")

        # Submit report
        result = env.step(AuditAction(
            action_type="submit_report",
            compliance_pass=tc["compliance_pass"],
        ))

        report = result.observation.result
        print(f"\n  FINAL SCORE: {report['final_score']}")
        print(f"  BREAKDOWN:   {json.dumps(report['breakdown'], indent=15)}")
        print(f"  DETAILS:     {json.dumps(report['details'], indent=15)}")

        # Show draft incident report
        incident = report.get("draft_incident_report", {})
        print(f"  VERDICT:     {incident.get('verdict', 'N/A')}")
        print(f"  CORRECT:     {incident.get('agent_verdict_correct', 'N/A')}")
        print(f"  ACTION:      {incident.get('recommended_action', 'N/A')}")
        print(f"  EFFICIENCY:  {incident.get('triage_efficiency_pct', 'N/A')}%")
        if incident.get("findings"):
            for f in incident["findings"]:
                print(f"    → {f['status']}: {f['finding']} ({f['severity']})")
        if incident.get("recommendations"):
            for r in incident["recommendations"]:
                print(f"    ★ {r}")
        print()

