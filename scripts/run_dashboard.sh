#!/usr/bin/env bash
set -e
cd "$HOME/2026 projects/ict-bot"
source .venv/bin/activate
PYTHONPATH=. python dashboard/app.py
