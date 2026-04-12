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

from openenv.core.env_server import create_app

from regtriage_openenv.environment import CallQAEnv
from regtriage_openenv.models import AuditAction, AuditObservation, AuditState


def create_regtriage_app():
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


# Create the app instance
app = create_regtriage_app()


def main(host: str = "0.0.0.0", port: int = 8000):
    """Entry point for running the server directly."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
