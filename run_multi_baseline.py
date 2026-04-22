"""
Multi-model, multi-run baseline runner for RegTriage.
Runs 3 models x 3 runs x 12 tasks and saves results to uniquely named JSON files.
Usage: uv run python run_multi_baseline.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Configuration
MODELS = [
    {"name": "google/gemma-4-31B-it", "display": "Gemma-4-31B-IT"},
    {"name": "google/gemma-4-26B-A4B-it", "display": "Gemma-4-26B-A4B-IT"},
    {"name": "Qwen/Qwen3.5-35B-A3B", "display": "Qwen-3.5-35B-A3B"},
]

NUM_RUNS = 3
RESULTS_DIR = Path("baseline_results_multi")

def run_single(model_name: str, run_idx: int, display_name: str) -> dict:
    """Run inference.py once with the given model."""
    env = os.environ.copy()
    env["MODEL_NAME"] = model_name
    env["API_BASE_URL"] = "https://router.huggingface.co/v1"
    # Ensure we use the HF token from .env
    load_dotenv = __import__("dotenv").load_dotenv
    load_dotenv()
    
    output_file = RESULTS_DIR / f"{display_name}_run{run_idx + 1}.json"
    
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"Running {display_name} — Run {run_idx + 1}/{NUM_RUNS}", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    
    start = time.time()
    proc = subprocess.run(
        [sys.executable, "inference.py"],
        cwd=Path(__file__).parent,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.time() - start
    
    # inference.py writes baseline_results.json
    default_output = Path("baseline_results.json")
    if default_output.exists():
        # Rename to per-model-per-run
        default_output.rename(output_file)
        print(f"Saved: {output_file} ({elapsed:.0f}s)", file=sys.stderr)
        
        # Load and return summary
        with open(output_file) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        return {
            "model": display_name,
            "run": run_idx + 1,
            "file": str(output_file),
            "overall_avg": summary.get("overall_avg", 0),
            "easy_avg": summary.get("easy_avg", 0),
            "medium_avg": summary.get("medium_avg", 0),
            "hard_avg": summary.get("hard_avg", 0),
            "total_time": summary.get("total_time", 0),
            "success": proc.returncode == 0,
        }
    else:
        print(f"ERROR: {default_output} not found. stderr:", file=sys.stderr)
        print(proc.stderr[-500:], file=sys.stderr)
        return {
            "model": display_name,
            "run": run_idx + 1,
            "file": None,
            "success": False,
            "error": "baseline_results.json not generated",
        }


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    
    all_summaries = []
    
    for model in MODELS:
        for run_idx in range(NUM_RUNS):
            summary = run_single(model["name"], run_idx, model["display"])
            all_summaries.append(summary)
            
            # Small delay between runs to avoid rate limits
            if run_idx < NUM_RUNS - 1:
                time.sleep(5)
    
    # Write master summary
    master_file = RESULTS_DIR / "_master_summary.json"
    with open(master_file, "w") as f:
        json.dump(all_summaries, f, indent=2)
    
    print(f"\n{'='*60}", file=sys.stderr)
    print("ALL RUNS COMPLETE", file=sys.stderr)
    print(f"{'='*60}", file=sys.stderr)
    for s in all_summaries:
        status = "OK" if s["success"] else "FAIL"
        print(f"[{status}] {s['model']} run{s['run']}: avg={s.get('overall_avg', 0):.3f}", file=sys.stderr)
    print(f"\nMaster summary: {master_file}", file=sys.stderr)


if __name__ == "__main__":
    main()
