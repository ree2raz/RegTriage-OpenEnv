"""Stub server/app.py for static validation compatibility.

The actual implementation is in regtriage_openenv.server.app.
This file exists to satisfy openenv static validator while maintaining
our package structure.
"""
import sys
from pathlib import Path

# Import from the actual package location
sys.path.insert(0, str(Path(__file__).parent.parent))

from regtriage_openenv.server.app import app, create_regtriage_app

def main():
    """Entry point for running the server."""
    from regtriage_openenv.server.app import main as _main
    return _main()

if __name__ == "__main__":
    main()
