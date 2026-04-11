#!/usr/bin/env python3
"""
demo.py — Local demonstration of RegTriage environment.

Quick start for local development without Docker or HF Spaces.
Shows the environment working with random actions.

Usage:
    uv run python demo.py
    uv run python demo.py --task call_006
"""

import argparse
import random
from typing import List

from regtriage_openenv import CallQAEnv, AuditAction


def random_agent_episode(env: CallQAEnv, task_id: str, verbose: bool = True) -> float:
    """Run one episode with random action selection."""
    obs = env.reset(task_id)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}")
        print(f"Difficulty: {env.current.get('difficulty', 'unknown')}")
        print(f"Budget: {env.total_budget} units")
        print(f"{'='*60}\n")
    
    step_count = 0
    total_reward = 0.0
    
    # Available actions
    tool_actions = [
        AuditAction(action_type="get_call_metadata"),
        AuditAction(action_type="get_sentiment_timeline"),
        AuditAction(action_type="read_transcript_chunk", start_turn=0, end_turn=2),
        AuditAction(action_type="analyze_turn", turn_index=1),
    ]
    
    while not env.done and step_count < 30:
        # Random action selection (with some bias toward useful actions)
        if step_count == 0:
            action = AuditAction(action_type="get_call_metadata")  # Always start with metadata
        elif step_count == 1:
            action = AuditAction(action_type="get_sentiment_timeline")  # Then sentiment
        elif random.random() < 0.3:
            # Sometimes flag a violation (random)
            violation_types = [
                "regulatory_disclosure_failure",
                "failed_escalation",
                "unauthorized_commitment",
                "incorrect_hold_procedure",
                "pii_exposure_risk",
                "churn_save_policy_breach",
            ]
            action = AuditAction(
                action_type="flag_violation",
                violation_type=random.choice(violation_types),
                violation_severity=random.choice(["high", "medium", "low"]),
                turn_index=random.randint(0, 5),
            )
        elif random.random() < 0.2:
            # Submit report
            action = AuditAction(action_type="submit_report", compliance_pass=False)
        else:
            # Random tool action
            action = random.choice(tool_actions)
        
        result = env.step(action)
        step_count += 1
        total_reward += result.reward
        
        if verbose:
            print(f"Step {step_count}: {action.action_type}")
            if result.observation.system_feedback:
                print(f"  Feedback: {result.observation.system_feedback[:80]}...")
            print(f"  Reward: {result.reward:.3f} | Budget: {env.budget_remaining}/{env.total_budget}")
        
        if result.done:
            break
    
    # Force submission if not done
    if not env.done:
        result = env.step(AuditAction(action_type="submit_report", compliance_pass=False))
        total_reward += result.reward
    
    # Get final score
    final_score = 0.0
    if isinstance(result.observation.result, dict) and "final_score" in result.observation.result:
        final_score = result.observation.result["final_score"]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Episode Complete!")
        print(f"Steps: {step_count}")
        print(f"Total Reward: {total_reward:.3f}")
        print(f"Final Score: {final_score:.3f}")
        print(f"{'='*60}\n")
    
    return final_score


def main():
    parser = argparse.ArgumentParser(description="RegTriage Environment Demo")
    parser.add_argument(
        "--task",
        type=str,
        default="call_001",
        help="Task ID to run (call_001 through call_012)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )
    
    args = parser.parse_args()
    
    # Create environment
    env = CallQAEnv()
    
    if args.list_tasks:
        print("\nAvailable Tasks:")
        for task in env.get_available_tasks():
            print(f"  {task['task_id']}: {task['difficulty']}")
        print()
        return
    
    # Run episodes
    scores = []
    for i in range(args.episodes):
        score = random_agent_episode(env, args.task, verbose=True)
        scores.append(score)
    
    if args.episodes > 1:
        print(f"\nSummary over {args.episodes} episodes:")
        print(f"  Mean Score: {sum(scores)/len(scores):.3f}")
        print(f"  Min Score: {min(scores):.3f}")
        print(f"  Max Score: {max(scores):.3f}")


if __name__ == "__main__":
    main()
