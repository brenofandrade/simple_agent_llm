#!/usr/bin/env bash
set -euo pipefail

# Ensure a stable pip version to avoid resolver AssertionError seen in pip 25.x
python -m pip install --upgrade "pip>=24.2,<25.0"

# Install project dependencies
python -m pip install -r requirements.txt
