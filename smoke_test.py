"""Smoke test: run call_006 once with Gemma 4 31B to verify API connectivity."""
import os
import sys

os.environ["MODEL_NAME"] = "google/gemma-4-31b-it"
os.environ["API_BASE_URL"] = "https://router.huggingface.co/v1"

# Only run call_006
os.environ["TASKS_TO_RUN"] = '["call_006"]'

from dotenv import load_dotenv
load_dotenv()

token = os.getenv("HF_TOKEN")
if not token:
    print("ERROR: HF_TOKEN not found", file=sys.stderr)
    sys.exit(1)

print(f"HF_TOKEN: {token[:8]}...{token[-4:]}", file=sys.stderr)
print(f"Model: {os.environ['MODEL_NAME']}", file=sys.stderr)
print(f"API: {os.environ['API_BASE_URL']}", file=sys.stderr)

# Patch TASKS_TO_RUN in inference.py by modifying the module before import
import inference
inference.TASKS_TO_RUN = [{"task_id": "call_006", "difficulty": "medium"}]

# Also patch MODEL_NAME
inference.MODEL_NAME = os.environ["MODEL_NAME"]
inference.client = inference.OpenAI(api_key=token, base_url=os.environ["API_BASE_URL"])

print("\nStarting smoke test (call_006, Gemma 4 31B)...", file=sys.stderr)
inference.main()
