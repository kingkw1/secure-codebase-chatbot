#!/bin/bash
# This needs to be run from the root project directory
# From there, it can be run with: bash tests/test_uvicorn.sh

# Set default values if not provided
PORT="${PORT:-9099}"
HOST="${HOST:-0.0.0.0}"

# Ensure script is run from the correct directory
cd "$(dirname "$0")/../pipelines"

# Run Uvicorn with the correct module path
../.venv/Scripts/python.exe -m uvicorn main:app --host "$HOST" --port "$PORT" --forwarded-allow-ips=\*
