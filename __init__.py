"""
RegTriage OpenEnv — Root package imports for backward compatibility.

This allows the old import style to work:
    from env import CallQAEnv
    from models import AuditAction

While also supporting the new package style:
    from regtriage_openenv import CallQAEnv
    from regtriage_openenv.models import AuditAction

Note: This is deprecated. Use regtriage_openenv package imports for new code.
"""

# Re-export from the main package for backward compatibility
from regtriage_openenv.environment import CallQAEnv
from regtriage_openenv.models import (
    AuditAction,
    AuditObservation,
    AuditState,
)

__all__ = [
    "CallQAEnv",
    "AuditAction",
    "AuditObservation",
    "AuditState",
]
