"""
Per-model baseline runner.
Usage: uv run python run_model_baseline.py <model_id> <display_name>
Runs 3 passes through all 12 tasks, saving results to unique JSON files.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if len(sys.argv) < 3:
    print("Usage: python run_model_baseline.py <model_id> <display_name>", file=sys.stderr)
    sys.exit(1)

MODEL_ID = sys.argv[1]
DISPLAY = sys.argv[2]
NUM_RUNS = 3
RESULTS_DIR = Path("baseline_results_multi")
RESULTS_DIR.mkdir(exist_ok=True)

env = os.environ.copy()
env["MODEL_NAME"] = MODEL_ID
env["API_BASE_URL"] = "https://router.huggingface.co/v1"

summaries = []

for run_idx in range(NUM_RUNS):
    output_file = RESULTS_DIR / f"{DISPLAY}_run{run_idx + 1}.json"
    env["OUTPUT_FILE"] = str(output_file)

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"[{DISPLAY}] Run {run_idx + 1}/{NUM_RUNS}", file=sys.stderr)
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

    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        summary = data.get("summary", {})
        summaries.append({
            "model": DISPLAY,
            "run": run_idx + 1,
            "file": str(output_file),
            "overall_avg": summary.get("overall_avg", 0),
            "easy_avg": summary.get("easy_avg", 0),
            "medium_avg": summary.get("medium_avg", 0),
            "hard_avg": summary.get("hard_avg", 0),
            "total_time": summary.get("total_time", 0),
            "elapsed_seconds": round(elapsed, 1),
            "success": proc.returncode == 0,
        })
        print(f"Saved: {output_file} ({elapsed:.0f}s)", file=sys.stderr)
    else:
        print(f"ERROR: {output_file} not found.", file=sys.stderr)
        print(proc.stderr[-800:], file=sys.stderr)
        summaries.append({
            "model": DISPLAY,
            "run": run_idx + 1,
            "success": False,
            "error": "output file not generated",
        })

    if run_idx < NUM_RUNS - 1:
        time.sleep(5)

# Write model summary
summary_file = RESULTS_DIR / f"{DISPLAY}_summary.json"
with open(summary_file, "w") as f:
    json.dump(summaries, f, indent=2)

print(f"\n[{DISPLAY}] Complete. Summary: {summary_file}", file=sys.stderr)
