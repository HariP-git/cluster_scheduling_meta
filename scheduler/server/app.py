# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Scheduler Environment.

This module creates an HTTP server that exposes the SchedulerEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 7860

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 7860 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from ..models import SchedulerAction, SchedulerObservation
    from .scheduler_environment import SchedulerEnvironment
except (ModuleNotFoundError, ImportError):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from models import SchedulerAction, SchedulerObservation
    from server.scheduler_environment import SchedulerEnvironment


# Create a singleton wrapper factory so HTTP REST calls (Swagger UI) maintain state sequentially
_shared_env = SchedulerEnvironment()

def get_shared_env():
    return _shared_env

# Create the app with web interface and README integration
app = create_app(
    get_shared_env,
    SchedulerAction,
    SchedulerObservation,
    env_name="scheduler",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)




def main():
    """
    Entry point for `uv run server`.

    Usage:
        uv run server                   # default host=0.0.0.0, port=7860
        uv run server --port 8001
        uv run server --host 127.0.0.1 --port 8080 --reload
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Scheduler Environment Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="Bind port (default: 7860)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    uvicorn.run(
        "scheduler.server.app:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
