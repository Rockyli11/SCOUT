#!/usr/bin/env bash
set -euo pipefail
python scout_runtime/run_scout.py --input scout_runtime/examples/prompts.jsonl --output outputs/predictions.jsonl
