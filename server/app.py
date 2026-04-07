"""
server.app — FastAPI server wrapping CallQAEnv for Docker/HuggingFace deployment.

Stateful HTTP server for multi-step episodes. Uses a singleton CallQAEnv
instance so reset/step calls maintain episode state across requests.

Standard OpenEnv endpoints:
    POST /reset         — Reset environment (optional task_id in body)
    POST /step          — Execute one agent action
    GET  /state         — Get current environment state
    GET  /health        — Health check (returns {"status": "healthy"})
    GET  /metadata      — Environment metadata
    GET  /schema        — Action/observation/state JSON schemas
    GET  /tasks         — List available tasks
    GET  /              — Root health check
"""

import sys
from pathlib import Path

# Ensure project root is importable (env.py lives at the project root)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from env import CallQAEnv, AuditAction, AuditObservation, AuditState

app = FastAPI(
    title="Call Center QA Auditor — OpenEnv",
    description="Contact center transcript compliance auditing environment",
    version="1.1.0",
)

env = CallQAEnv()


# ── Request Models ────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = None


# ── Standard OpenEnv Endpoints ────────────────────────────────

@app.get("/health")
def health():
    """OpenEnv standard health check."""
    return {"status": "healthy"}


@app.get("/metadata")
def metadata():
    """OpenEnv standard metadata endpoint."""
    return {
        "name": "regtriage",
        "description": (
            "An OpenEnv environment for training RL agents on financial services "
            "regulatory compliance auditing and revenue leakage detection. Produces "
            "Draft Incident Reports for human supervisor sign-off."
        ),
        "version": "2.0.0",
        "tasks": env.get_available_tasks(),
    }


@app.get("/schema")
def schema():
    """OpenEnv standard schema endpoint — action, observation, state JSON schemas."""
    return {
        "action": AuditAction.model_json_schema(),
        "observation": AuditObservation.model_json_schema(),
        "state": AuditState.model_json_schema(),
    }


@app.post("/mcp")
def mcp(request: dict = {}):
    """Minimal MCP JSON-RPC endpoint for OpenEnv runtime compliance."""
    method = request.get("method", "")
    request_id = request.get("id")
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}. This environment uses HTTP endpoints, not MCP.",
        },
    }


# ── Core Simulation Endpoints ────────────────────────────────

@app.get("/")
def root():
    """Root health check — returns 200 if server is alive."""
    return {"status": "ok", "environment": "regtriage", "version": "2.0.0"}


@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    """Reset the environment. Optionally specify a task_id."""
    obs = env.reset(request.task_id)
    return {
        "observation": obs.model_dump(),
        "reward": 0.0,
        "done": False,
        "info": {"available_tasks": [t["task_id"] for t in env.get_available_tasks()]},
    }


@app.post("/step")
def step(action: AuditAction):
    """Execute one agent action and return the result."""
    result = env.step(action)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
        "info": result.info,
    }


@app.get("/state")
def state():
    """Get current environment state."""
    return env.state().model_dump()


@app.get("/tasks")
def list_tasks():
    """List all available tasks with difficulty levels."""
    return {"tasks": env.get_available_tasks()}


# ── Entry Point ──────────────────────────────────────────────

def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution.

    Supports:
        uv run server
        python -m server.app
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
