"""OpenEnv client for the RegTriage Environment.

Usage (remote / Docker / HF Spaces)::

    from regtriage_openenv.client import RegTriageEnv
    from regtriage_openenv.models import AuditAction

    # Connect to a running server (local, Docker, or HF Space)
    client = RegTriageEnv(base_url="http://localhost:7860")

    with client:
        result = client.reset("call_001")
        print(result.observation.result)

        result = client.step(AuditAction(action_type="get_call_metadata"))
        print(result.observation.budget_remaining_pct)

    # Or use Docker image directly
    client = RegTriageEnv.from_docker_image("regtriage:2.0.1")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from regtriage_openenv.models import AuditAction, AuditObservation, AuditState

# ---------------------------------------------------------------------------
# OpenEnv EnvClient (with fallback stub)
# ---------------------------------------------------------------------------

try:
    from openenv.core.env_client import EnvClient
    from openenv.core.client_types import StepResult

    class RegTriageEnv(EnvClient[AuditAction, AuditObservation, AuditState]):
        """Typed WebSocket/HTTP client for the RegTriage environment.
        
        Supports connection to running OpenEnv servers via HTTP or WebSocket.
        """

        def _step_payload(self, action: AuditAction) -> Dict[str, Any]:
            """Convert action to payload for HTTP/WebSocket transmission."""
            return action.model_dump(exclude_none=True)

        def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AuditObservation]:
            """Parse server response into StepResult."""
            obs_data = payload.get("observation", {})
            
            # Handle nested observation structure
            if "result" in obs_data:
                result = obs_data["result"]
                checklist = obs_data.get("checklist", {})
                feedback = obs_data.get("system_feedback", "")
            else:
                # Flat structure fallback
                result = obs_data
                checklist = {}
                feedback = ""
            
            obs = AuditObservation(
                result=result,
                checklist=checklist,
                system_feedback=feedback,
            )
            
            return StepResult(
                observation=obs,
                reward=payload.get("reward", 0.0),
                done=payload.get("done", False),
            )

        def _parse_state(self, payload: Dict[str, Any]) -> AuditState:
            """Parse state response into AuditState."""
            return AuditState(**payload)

except ImportError:
    # When openenv-core is not installed, provide a thin HTTP-based client
    # that works with the plain FastAPI server. This keeps the project
    # functional for local experimentation without Docker.
    
    import json
    import urllib.request
    from dataclasses import dataclass

    @dataclass
    class _StepResult:
        """Minimal stand-in for openenv.core.client_types.StepResult."""
        observation: AuditObservation
        reward: float
        done: bool

    class RegTriageEnv:  # type: ignore[no-redef]
        """Lightweight HTTP client that talks to the FastAPI server.
        
        Works without openenv-core installed for local development.
        """

        def __init__(self, base_url: str = "http://localhost:7860"):
            self.base_url = base_url.rstrip("/")
            self._closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()

        def close(self):
            """Close the client connection."""
            self._closed = True

        def reset(self, task_id: Optional[str] = None, seed: Optional[int] = None) -> _StepResult:
            """Reset the environment."""
            body = json.dumps({"task_id": task_id, "seed": seed}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/reset",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            
            obs_data = resp.get("observation", resp)
            obs = AuditObservation(
                result=obs_data.get("result"),
                checklist=obs_data.get("checklist", {}),
                system_feedback=obs_data.get("system_feedback", ""),
            )
            return _StepResult(obs, resp.get("reward", 0.0), resp.get("done", False))

        def step(self, action: AuditAction) -> _StepResult:
            """Execute one step in the environment."""
            body = json.dumps({"action": action.model_dump(exclude_none=True)}).encode()
            req = urllib.request.Request(
                f"{self.base_url}/step",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = json.loads(urllib.request.urlopen(req).read())
            
            obs_data = resp.get("observation", resp)
            obs = AuditObservation(
                result=obs_data.get("result"),
                checklist=obs_data.get("checklist", {}),
                system_feedback=obs_data.get("system_feedback", ""),
            )
            return _StepResult(obs, resp.get("reward", 0.0), resp.get("done", False))

        def state(self) -> AuditState:
            """Get current environment state."""
            resp = json.loads(urllib.request.urlopen(f"{self.base_url}/state").read())
            return AuditState(**resp)

        @classmethod
        def from_docker_image(cls, image: str, **kwargs):
            """Not available without openenv-core."""
            raise NotImplementedError(
                "from_docker_image requires openenv-core. "
                "Install it with: pip install 'openenv-core[core]>=0.2.1'"
            )

        @classmethod
        def from_hub(cls, repo_id: str, **kwargs):
            """Not available without openenv-core."""
            raise NotImplementedError(
                "from_hub requires openenv-core. "
                "Install it with: pip install 'openenv-core[core]>=0.2.1'"
            )
