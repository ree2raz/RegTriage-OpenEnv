"""server.app — FastAPI server for RegTriage using OpenEnv create_app().

This uses the standard OpenEnv server pattern with create_app() for full
compliance with the OpenEnv specification.
"""

import sys
from pathlib import Path

# Ensure project root is importable
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from regtriage_openenv.environment import CallQAEnv
from regtriage_openenv.models import AuditAction, AuditObservation, AuditState

# Try to use OpenEnv's create_app if available
try:
    from openenv.core.env_server import create_app
    
    def create_regtriage_app() -> FastAPI:
        """Create FastAPI app using OpenEnv's create_app."""
        # create_app expects env as a factory (callable)
        def env_factory():
            return CallQAEnv()
        
        app = create_app(
            env=env_factory,
            action_cls=AuditAction,
            observation_cls=AuditObservation,
            env_name="regtriage",
        )
        
        return app

except ImportError:
    # Fallback: hand-rolled FastAPI when openenv-core not installed
    
    def create_regtriage_app() -> FastAPI:
        """Create FastAPI app manually (fallback)."""
        env = CallQAEnv()
        
        app = FastAPI(
            title="RegTriage — Regulatory Compliance Auditing Environment",
            description="OpenEnv environment for training RL agents on financial services compliance",
            version="2.0.1",
        )
        
        class ResetRequest(BaseModel):
            task_id: Optional[str] = None
            seed: Optional[int] = None
        
        @app.get("/health")
        def health():
            return {"status": "healthy", "service": "regtriage"}
        
        @app.get("/metadata")
        def metadata():
            return {
                "name": "regtriage",
                "description": "Financial services regulatory compliance auditing environment",
                "version": "2.0.1",
                "tasks": env.get_available_tasks(),
            }
        
        @app.get("/schema")
        def schema():
            return {
                "action": AuditAction.model_json_schema(),
                "observation": AuditObservation.model_json_schema(),
                "state": AuditState.model_json_schema(),
            }
        
        @app.get("/")
        def root():
            return {"status": "ok", "environment": "regtriage", "version": "2.0.1"}
        
        @app.post("/reset")
        def reset(request: ResetRequest = ResetRequest()):
            obs = env.reset(request.task_id)
            return {
                "observation": obs.model_dump(),
                "reward": 0.0,
                "done": False,
                "info": {"available_tasks": env.get_available_tasks()},
            }
        
        @app.post("/step")
        def step(action: AuditAction):
            result = env.step(action)
            return {
                "observation": result.observation.model_dump(),
                "reward": result.reward,
                "done": result.done,
                "info": result.info,
            }
        
        @app.get("/state")
        def state():
            return env.state().model_dump()
        
        @app.get("/tasks")
        def list_tasks():
            return {"tasks": env.get_available_tasks()}
        
        return app

# Create the app instance
app = create_regtriage_app()

def main(host: str = "0.0.0.0", port: int = 7860):
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
