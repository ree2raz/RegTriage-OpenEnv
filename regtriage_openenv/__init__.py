"""RegTriage OpenEnv — Financial Services Regulatory Compliance Auditing Environment.

Two ways to use this package:

1. **Local** (development path — no extra deps):
   from regtriage_openenv import CallQAEnv
   from regtriage_openenv.models import AuditAction
   env = CallQAEnv()
   obs = env.reset("call_001")

2. **OpenEnv** (deployment path — needs openenv-core):
   from regtriage_openenv.client import RegTriageEnv
   from regtriage_openenv.models import AuditAction
   client = RegTriageEnv.from_docker_image("regtriage:2.0.1")
   # or
   client = RegTriageEnv(base_url="https://ree2raz-regtriage-openenv.hf.space")
   client.reset("call_001")
   client.step(AuditAction(action_type="get_call_metadata"))

OpenEnv-compatible environment for training RL agents to audit financial
services contact center transcripts for regulatory compliance violations.
"""

from regtriage_openenv.environment import CallQAEnv
from regtriage_openenv.models import (
    AuditAction,
    AuditObservation,
    AuditState,
    StepResult,
)

__version__ = "2.0.1"

__all__ = [
    "CallQAEnv",
    "AuditAction",
    "AuditObservation",
    "AuditState",
    "StepResult",
]
