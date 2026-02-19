#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${ROOT_DIR}/dist"
TS="$(date -u +%Y%m%d-%H%M%S)"
ZIP_PATH="${OUT_DIR}/btc_perp_research_llm_review_${TS}.zip"

mkdir -p "${OUT_DIR}"

cd "${ROOT_DIR}"

# Build a clean review zip: code/config/docs/scripts only, no heavy data or run outputs.
zip -r "${ZIP_PATH}" \
  . \
  -x ".git/*" \
  -x ".venv/*" \
  -x "__pycache__/" \
  -x "*/__pycache__/" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "*/.DS_Store" \
  -x ".pytest_cache" \
  -x ".pytest_cache/" \
  -x ".pytest_cache/*" \
  -x "research/data_cache/*" \
  -x "research/artefacts/logs/*" \
  -x "research/artefacts/runs/*" \
  -x "*.parquet" \
  -x "*.db" \
  -x "*.sqlite" \
  -x "*.sqlite3" \
  -x "dist/*"

echo "${ZIP_PATH}"
