"""
grading.py — Scoring logic for the Call Center QA Environment.

Pure functions with no environment state dependency. The environment
calls these with the relevant data; they return scores.

Scoring Components (on submit_report):
  1. Compliance verdict   (0.20) — Binary: agent's call matches ground truth
  2. Violation detection  (0.60) — Severity-weighted F1 score
  3. Efficiency bonus     (0.20) — budget_remaining / total_budget
  - False positive penalty: weighted by claimed severity
  - Severity calibration: +0.02 per exact severity match
  - Auto-fail cap: if any high-severity violation missed, max score = 0.30

The severity-weighted F1 treats not all violations equally:
   - high   = 3× weight (e.g., regulatory disclosure failure, failed escalation)
   - medium = 2× weight (e.g., unauthorized commitment, PII exposure)
   - low    = 1× weight (e.g., minor hold procedure violation)

This mirrors real-world QA where missing a regulatory violation is
far more consequential than missing a procedural issue.
"""

# Severity weights used in F1 calculation
SEVERITY_WEIGHTS = {"high": 3.0, "medium": 2.0, "low": 1.0}

# False positive penalty weights — claiming a high-severity FP is more costly
FP_PENALTY_WEIGHTS = {"high": 0.10, "medium": 0.05, "low": 0.03}


def compute_violation_f1(
    gt_violations: list[dict],
    flagged_violations: list[dict],
) -> tuple[float, int, int, int, bool, int]:
    """
    Compute severity-weighted F1 between agent flags and ground truth.

    Matching is on violation TYPE only (not turn_index). In real-world QA,
    identifying the violation category is what matters — agents may reasonably
    cite different turns for the same multi-turn violation.

    Each match is weighted by severity: high=3, medium=2, low=1.

    Also tracks severity calibration: how many matched violations have
    agent-assigned severity matching ground truth severity exactly.

    Args:
        gt_violations:     Ground truth violation list from transcript data
        flagged_violations: Agent's flagged violations during the episode

    Returns:
        (weighted_f1_score, true_positives, false_positives, false_negatives,
         missed_any_high, severity_matches)
    """
    # Build type-based multisets for greedy matching
    gt_types = [v["type"] for v in gt_violations]
    agent_types = [v["type"] for v in flagged_violations]

    # Severity lookup: type → severity (from ground truth)
    gt_severity = {}
    for v in gt_violations:
        gt_severity[v["type"]] = v.get("severity", "low")

    # Agent severity lookup for matched violations
    agent_severity_by_type = {}
    for v in flagged_violations:
        agent_severity_by_type[v["type"]] = v.get("severity", "low")

    # Greedy matching: for each agent flag, consume one ground truth of same type
    gt_remaining = list(gt_types)  # mutable copy
    tp_types = []

    for a_type in agent_types:
        if a_type in gt_remaining:
            gt_remaining.remove(a_type)  # consume one match
            tp_types.append(a_type)

    tp = len(tp_types)
    fp = len(agent_types) - tp
    fn = len(gt_remaining)  # unmatched ground truth

    # Severity calibration: count exact severity matches
    severity_matches = sum(
        1 for t in tp_types
        if agent_severity_by_type.get(t) == gt_severity.get(t)
    )

    # Weighted precision and recall
    tp_weight = sum(SEVERITY_WEIGHTS.get(gt_severity.get(t, "low"), 1.0) for t in tp_types)
    total_gt_weight = sum(SEVERITY_WEIGHTS.get(v.get("severity", "low"), 1.0) for v in gt_violations)

    # FP weight now based on agent's CLAIMED severity
    fp_agents = [v for v in flagged_violations if v["type"] not in gt_types or
                 agent_types.count(v["type"]) > gt_types.count(v["type"])]
    total_agent_weight = tp_weight + len([v for v in flagged_violations if v["type"] not in set(gt_types)])

    if total_agent_weight == 0 and tp_weight == 0:
        # No flags at all
        if total_gt_weight == 0:
            weighted_f1 = 1.0  # No violations expected, none flagged = perfect
        else:
            weighted_f1 = 0.0
    elif total_gt_weight == 0:
        weighted_f1 = 0.0  # Ground truth has no violations but agent flagged some
    else:
        total_flagged_weight = tp_weight + fp  # FPs count as 1.0 each for precision calc
        if total_flagged_weight == 0:
            precision = 0.0
        else:
            precision = tp_weight / total_flagged_weight
        recall = tp_weight / total_gt_weight
        if (precision + recall) > 0:
            weighted_f1 = 2 * precision * recall / (precision + recall)
        else:
            weighted_f1 = 0.0

    # Scale to the violation detection component weight (0.60)
    violation_score = 0.6 * weighted_f1

    # Check if any high-severity ground truth violation was missed entirely
    high_gt_types = {v["type"] for v in gt_violations if v.get("severity") == "high"}
    high_caught = high_gt_types & set(agent_types)
    missed_high = len(high_gt_types) > 0 and len(high_caught) == 0

    return violation_score, tp, fp, fn, missed_high, severity_matches


def build_draft_incident_report(
    gt_violations: list[dict],
    flagged_violations: list[dict],
    agent_compliance: bool,
    gt_compliance: bool,
    tp_types: list[str],
    budget_remaining: int,
    total_budget: int,
) -> dict:
    """
    Build a Draft Incident Report for human QA supervisor review.

    This is the audit trail artifact that reframes AI output as
    human-augmented triage — the AI agent acts as an automated scout,
    and the human supervisor makes the final determination.

    Used by:
    1. QA supervisors for final sign-off on flagged incidents
    2. QA managers to identify systemic compliance gaps
    3. Compliance officers as regulatory documentation
    4. RL training loop as structured reward signal

    Generated deterministically from grading data, not from an LLM.
    """
    # Verdict rationale
    total_gt = len(gt_violations)
    high_count = sum(1 for v in gt_violations if v.get("severity") == "high")
    medium_count = sum(1 for v in gt_violations if v.get("severity") == "medium")
    low_count = sum(1 for v in gt_violations if v.get("severity") == "low")

    if gt_compliance:
        rationale = "No compliance violations detected in this call."
    else:
        parts = []
        if high_count:
            parts.append(f"{high_count} high-severity")
        if medium_count:
            parts.append(f"{medium_count} medium-severity")
        if low_count:
            parts.append(f"{low_count} low-severity")
        rationale = f"{total_gt} violation(s) identified: {', '.join(parts)}."

    # Per-violation findings
    flagged_types = [v["type"] for v in flagged_violations]
    findings = []

    for v in gt_violations:
        detected = v["type"] in flagged_types
        findings.append({
            "finding": v["type"].replace("_", " ").title(),
            "severity": v.get("severity", "low").upper(),
            "detected": detected,
            "detail": v.get("description", "No description available."),
            "status": "DETECTED" if detected else "MISSED",
        })

    # False positives (agent flagged something not in ground truth)
    gt_type_set = {v["type"] for v in gt_violations}
    for v in flagged_violations:
        if v["type"] not in gt_type_set:
            findings.append({
                "finding": v["type"].replace("_", " ").title(),
                "severity": v.get("severity", "low").upper(),
                "detected": False,
                "detail": "Agent flagged this violation but it does not exist in ground truth.",
                "status": "FALSE POSITIVE",
            })

    # Recommendations
    recommendations = []
    missed_types = [v["type"] for v in gt_violations if v["type"] not in flagged_types]

    if missed_types:
        for mt in missed_types:
            friendly = mt.replace("_", " ").title()
            recommendations.append(f"Review detection methodology for '{friendly}' violations.")

    if not agent_compliance and gt_compliance:
        recommendations.append("Calibrate compliance verdict — this call was actually compliant.")
    elif agent_compliance and not gt_compliance:
        recommendations.append("Tighten compliance threshold — violations were present but verdict was 'pass'.")

    fp_types = [v["type"] for v in flagged_violations if v["type"] not in gt_type_set]
    if fp_types:
        recommendations.append("Reduce false positive rate — review flagging criteria for precision.")

    if not recommendations:
        recommendations.append("Audit performance meets quality standards. No corrective action needed.")

    # Triage efficiency
    triage_efficiency = round(budget_remaining / total_budget * 100) if total_budget > 0 else 0

    # Recommended action for human supervisor
    if not gt_compliance and high_count > 0:
        recommended_action = "ESCALATE — High-severity regulatory violation(s) require immediate QA supervisor review and sign-off."
    elif not gt_compliance:
        recommended_action = "REVIEW — Medium/low-severity findings require QA supervisor review."
    else:
        recommended_action = "ARCHIVE — No violations detected. Safe to archive without further review."

    # Estimated human review time (minutes)
    # Clean calls: 0 min, flagged: 5 min base + 3 min per violation
    estimated_review_minutes = 0 if gt_compliance else (5 + 3 * total_gt)

    return {
        "verdict": "PASS" if gt_compliance else "FAIL",
        "agent_verdict_correct": agent_compliance == gt_compliance,
        "verdict_rationale": rationale,
        "findings": findings,
        "recommendations": recommendations,
        "recommended_action": recommended_action,
        "triage_efficiency_pct": triage_efficiency,
        "estimated_human_review_minutes": estimated_review_minutes,
    }


def grade_report(
    gt_violations: list[dict],
    gt_compliance: bool,
    flagged_violations: list[dict],
    agent_compliance: bool,
    budget_remaining: int,
    total_budget: int,
) -> tuple[dict, float]:
    """
    Grade the agent's complete audit report.

    Called once when the agent submits their report (or when budget
    exhaustion forces auto-submission).

    Args:
        gt_violations:      Ground truth violations from transcript data
        gt_compliance:      Ground truth compliance pass/fail
        flagged_violations: Agent's flagged violations
        agent_compliance:   Agent's compliance verdict
        budget_remaining:   Remaining compute budget units
        total_budget:       Total compute budget allocated

    Returns:
        (result_dict, final_score) where result_dict contains breakdown,
        details, and structured compliance summary.
    """
    # ── Component 1: Compliance call (0.20) ──────────
    compliance_score = 0.2 if (agent_compliance == gt_compliance) else 0.0

    # ── Component 2: Severity-weighted violation F1 (0.60) ──
    violation_score, tp, fp, fn, missed_high, severity_matches = compute_violation_f1(
        gt_violations, flagged_violations
    )

    # ── Component 3: Efficiency bonus (0.20) ─────────
    # Budget-based: reward efficient use of compute resources
    if total_budget > 0:
        efficiency = budget_remaining / total_budget
    else:
        efficiency = 0.0
    efficiency_score = 0.2 * efficiency

    # ── Bonus: Severity calibration ──────────────────
    # +0.02 per violation where agent's severity matches ground truth exactly
    calibration_bonus = 0.02 * severity_matches

    # ── Penalty: Weighted false positives ────────────
    # FP penalty weighted by agent's claimed severity
    fp_penalty = 0.0
    gt_type_set = {v["type"] for v in gt_violations}
    for v in flagged_violations:
        if v["type"] not in gt_type_set:
            claimed_sev = v.get("severity", "low")
            fp_penalty += FP_PENALTY_WEIGHTS.get(claimed_sev, 0.05)

    # ── Raw total ────────────────────────────────────
    raw_total = compliance_score + violation_score + efficiency_score + calibration_bonus - fp_penalty

    # ── Auto-fail cap ────────────────────────────────
    # If ground truth has any high-severity violation and agent missed ALL of them,
    # cap the maximum score at 0.30 (models real-world QA auto-fail behavior)
    auto_fail_applied = False
    if missed_high:
        raw_total = min(raw_total, 0.30)
        auto_fail_applied = True

    total = max(0.0, min(1.0, raw_total))

    # ── Build structured compliance summary ──────────
    flagged_types = [v["type"] for v in flagged_violations]
    gt_remaining = [v["type"] for v in gt_violations]
    tp_types_list = []
    for ft in flagged_types:
        if ft in gt_remaining:
            gt_remaining.remove(ft)
            tp_types_list.append(ft)

    draft_incident_report = build_draft_incident_report(
        gt_violations=gt_violations,
        flagged_violations=flagged_violations,
        agent_compliance=agent_compliance,
        gt_compliance=gt_compliance,
        tp_types=tp_types_list,
        budget_remaining=budget_remaining,
        total_budget=total_budget,
    )

    result = {
        "final_score": round(total, 3),
        "breakdown": {
            "compliance_correct": round(compliance_score, 3),
            "violation_f1_weighted": round(violation_score, 3),
            "efficiency_bonus": round(efficiency_score, 3),
            "severity_calibration_bonus": round(calibration_bonus, 3),
            "false_positive_penalty": round(fp_penalty, 3),
            "auto_fail_applied": auto_fail_applied,
        },
        "details": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "severity_matches": severity_matches,
            "ground_truth_count": len(gt_violations),
            "agent_flagged_count": len(flagged_violations),
            "budget_remaining": budget_remaining,
            "total_budget": total_budget,
            "missed_high_severity": missed_high,
        },
        "draft_incident_report": draft_incident_report,
    }
    return result, total
