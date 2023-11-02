#!/bin/bash

# Use the provided PYTHON_PATH or default to the system's Python
PYTHON_PATH="${PYTHON_PATH:-$(which python)}"

# Exit immediately if a command exits with a non-zero status.
set -e

$PYTHON_PATH -m autoflake "$@"
$PYTHON_PATH -m black "$@"
$PYTHON_PATH -m isort "$@"
