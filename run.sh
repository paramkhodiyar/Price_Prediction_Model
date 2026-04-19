#!/bin/bash
# ValoraAI — local development launcher
# Usage: ./run.sh [port]
# Requires Python 3.12 (falls back to 3.11, 3.10, 3.9)

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [ ! -f ".env" ]; then
  echo "WARNING: .env file not found. Some API features (like Groq) may fail at runtime."
fi

PORT="${1:-8501}"
VENV_DIR="./venv"

# ── Find a compatible Python (3.9–3.12; NOT 3.13/3.14 — protobuf breaks) ────
find_python() {
  for candidate in python3.12 python3.11 python3.10 python3.9 python3 python /usr/bin/python3; do
    if command -v "$candidate" &>/dev/null; then
      if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info[:2] in [(3,9), (3,10), (3,11), (3,12)] else 1)' &>/dev/null; then
        echo "$candidate"
        return
      fi
    fi
  done
  echo ""
}

PYTHON=$(find_python)
if [ -z "$PYTHON" ]; then
  echo "ERROR: No compatible Python (3.9–3.12) found."
  echo "Install Python 3.12 from https://www.python.org/downloads/"
  exit 1
fi

echo "Using $PYTHON ($(${PYTHON} --version 2>&1))"

# ── Create venv if missing ────────────────────────────────────────────────────
if [ ! -f "$VENV_DIR/bin/python" ]; then
  echo "Creating virtual environment ..."
  $PYTHON -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
PY="$VENV_DIR/bin/python"

# ── Install / sync dependencies ───────────────────────────────────────────────
echo "Upgrading pip, setuptools, and wheel..."
$PIP install --quiet --upgrade pip setuptools wheel

echo "Installing dependencies (this may take ~1-2 min first time)..."
$PIP install --quiet -r deployment/requirements.txt

# ── Pre-build FAISS vector store if not present ───────────────────────────────
if [ ! -d "./rag_store" ]; then
  echo "Building RAG vector store (first-time, ~20 s) ..."
  $PY -m rag.setup
  echo "RAG store ready."
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "Starting ValoraAI on http://localhost:${PORT}"
echo "Press Ctrl+C to stop."
echo ""

$PY -m streamlit run app/app.py \
  --server.port "$PORT" \
  --server.headless false \
  --server.address 0.0.0.0
