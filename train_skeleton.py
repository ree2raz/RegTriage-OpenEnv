#!/usr/bin/env python3
"""
train_skeleton.py — Minimal TRL training example for RegTriage.

This demonstrates connecting TRL's GRPOTrainer to the RegTriage environment.
The skeleton shows the wiring — training convergence requires hyperparameter tuning.

Prerequisites:
    pip install trl torch transformers

Usage:
    # Terminal 1: Start environment server
    uv run python -m regtriage_openenv.server.app

    # Terminal 2: Run training
    uv run python train_skeleton.py --env-url http://localhost:7860

Or use local environment (no server):
    uv run python train_skeleton.py --local

The goal is to train a 7B model to beat zero-shot 72B performance on the
Hero Agent trap (call_006) by learning to distinguish:
    - churn_save_policy_breach (P&L leak) vs
    - unauthorized_commitment (legal liability)
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Check for TRL
try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError:
    print("Error: trl not installed. Install with: pip install trl")
    sys.exit(1)

# RegTriage imports
from regtriage_openenv.models import AuditAction, AuditObservation


def create_system_prompt() -> str:
    """Create the system prompt for the compliance auditing agent."""
    return """You are an expert Quality Assurance (QA) supervisor auditing call transcripts for regulatory compliance.

Your goal is to identify violations efficiently under a compute budget. Use tools strategically:
- get_call_metadata (5 units): Start here for triage
- get_sentiment_timeline (5 units): Find emotional hotspots
- read_transcript_chunk (3 units/turn): Read specific sections
- analyze_turn (10 units): Deep analysis with policy rubric
- flag_violation (2 units): Record findings
- submit_report (0 units): Final compliance verdict

Key challenge: Distinguish churn_save_policy_breach (giving unauthorized discounts) from 
unauthorized_commitment (promising outcomes). The customer may be happy in both cases!

Always use tools via function calls. Be efficient with budget."""


def format_observation_for_llm(obs: AuditObservation, step: int) -> str:
    """Convert observation to text for LLM context."""
    lines = [f"Step {step}:"]
    
    if obs.result:
        lines.append(f"Result: {json.dumps(obs.result, indent=2)[:500]}")
    
    if obs.checklist:
        budget_pct = obs.checklist.get("budget_remaining_pct", 0)
        lines.append(f"Budget remaining: {budget_pct}%")
    
    if obs.system_feedback:
        lines.append(f"Feedback: {obs.system_feedback[:200]}")
    
    return "\n".join(lines)


class RegTriageRewardFunction:
    """Reward function wrapper for TRL training."""
    
    def __init__(self, env_url: Optional[str] = None, use_local: bool = False):
        self.env_url = env_url
        self.use_local = use_local
        self.env = None
        
    def _get_env(self):
        """Lazy initialization of environment."""
        if self.env is None:
            if self.use_local:
                from regtriage_openenv import CallQAEnv
                self.env = CallQAEnv()
            else:
                from regtriage_openenv.client import RegTriageEnv
                self.env = RegTriageEnv(base_url=self.env_url or "http://localhost:7860")
                self.env.__enter__()
        return self.env
    
    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        """
        Compute rewards for a batch of completions.
        
        Each completion should be a sequence of tool calls ending with submit_report.
        We parse the completion, execute actions, and return the final score.
        """
        rewards = []
        env = self._get_env()
        
        for prompt, completion in zip(prompts, completions):
            try:
                # Parse completion to extract actions
                # Expected format: tool_name(param=value, ...) or similar
                reward = self._evaluate_completion(env, prompt, completion)
                rewards.append(reward)
            except Exception as e:
                print(f"Error evaluating completion: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def _evaluate_completion(self, env, prompt: str, completion: str) -> float:
        """Evaluate a single completion by running it in the environment."""
        # Reset with a random task (or fixed task for debugging)
        import random
        task_id = f"call_{random.randint(1, 12):03d}"
        
        result = env.reset(task_id)
        total_reward = 0.0
        
        # Parse actions from completion (simplified)
        # In practice, you'd parse structured output or use constrained generation
        actions = self._parse_actions(completion)
        
        for action in actions:
            step_result = env.step(action)
            total_reward += step_result.reward
            
            if step_result.done:
                # Extract final score from result
                if isinstance(step_result.observation.result, dict):
                    final_score = step_result.observation.result.get("final_score", 0.0)
                    return final_score
                return total_reward
        
        # If no submit_report, force one
        result = env.step(AuditAction(action_type="submit_report", compliance_pass=False))
        total_reward += result.reward
        
        if isinstance(result.observation.result, dict):
            return result.observation.result.get("final_score", total_reward)
        return total_reward
    
    def _parse_actions(self, completion: str) -> List[AuditAction]:
        """Parse tool calls from LLM completion text."""
        actions = []
        
        # Simple regex-based parsing (production would use structured output)
        import re
        
        # Look for patterns like: get_call_metadata() or flag_violation(type="...")
        tool_pattern = r'(\w+)\(([^)]*)\)'
        matches = re.findall(tool_pattern, completion)
        
        for tool_name, params_str in matches:
            try:
                # Parse parameters
                params = {}
                if params_str:
                    for param in params_str.split(','):
                        if '=' in param:
                            key, val = param.split('=', 1)
                            key = key.strip()
                            val = val.strip().strip('"\'')
                            # Try to convert to int/bool
                            if val.isdigit():
                                val = int(val)
                            elif val.lower() == 'true':
                                val = True
                            elif val.lower() == 'false':
                                val = False
                            params[key] = val
                
                action = AuditAction(action_type=tool_name, **params)
                actions.append(action)
            except Exception:
                continue
        
        return actions


def main():
    parser = argparse.ArgumentParser(description="Train RL agent on RegTriage")
    parser.add_argument(
        "--env-url",
        type=str,
        default="http://localhost:7860",
        help="URL of running RegTriage environment server",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local environment instead of HTTP server",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model to train (7B recommended for faster iteration)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./regtriage-grpo-output",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num-train-steps",
        type=int,
        default=100,
        help="Number of training steps",
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("RegTriage RL Training Skeleton")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Environment: {'local' if args.local else args.env_url}")
    print(f"Output: {args.output_dir}")
    print("="*60)
    
    # Load model and tokenizer
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    
    # Create reward function
    reward_fn = RegTriageRewardFunction(
        env_url=args.env_url,
        use_local=args.local,
    )
    
    # Training configuration
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-5,
        logging_steps=10,
        save_steps=50,
        max_prompt_length=2048,
        max_completion_length=1024,
        # GRPO specific
        num_generations=4,  # Number of completions per prompt
        temperature=0.7,
    )
    
    # Training data (simplified — in practice use diverse prompts)
    train_texts = [
        create_system_prompt() + "\n\nAudit this call transcript.",
    ] * 10  # Repeat for batch
    
    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=[{"text": t} for t in train_texts],
        reward_funcs=reward_fn,
    )
    
    print("\nStarting training...")
    print("(This skeleton demonstrates the wiring. Convergence requires tuning.)")
    
    try:
        trainer.train()
        print(f"\nTraining complete! Checkpoints saved to {args.output_dir}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Cleanup
        if hasattr(reward_fn.env, '__exit__'):
            reward_fn.env.__exit__(None, None, None)


if __name__ == "__main__":
    main()
