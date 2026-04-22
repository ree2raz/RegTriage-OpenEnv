"""Re-run Qwen 3.5-35B-A3B run 3 only."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
DISPLAY = "Qwen-3.5-35B-A3B"
RESULTS_DIR = Path("baseline_results_multi")

env = os.environ.copy()
env["MODEL_NAME"] = MODEL_ID
env["API_BASE_URL"] = "https://router.huggingface.co/v1"

output_file = RESULTS_DIR / f"{DISPLAY}_run3.json"
env["OUTPUT_FILE"] = str(output_file)

print(f"Running {DISPLAY} — Run 3/3 (re-run)", file=sys.stderr)

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
    print(f"Saved: {output_file} ({elapsed:.0f}s)", file=sys.stderr)
    print(f"Overall avg: {summary.get('overall_avg', 0):.3f}", file=sys.stderr)
else:
    print(f"ERROR: {output_file} not found.", file=sys.stderr)
    print(proc.stderr[-800:], file=sys.stderr)
