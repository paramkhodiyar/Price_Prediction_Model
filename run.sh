#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/venv"
PYTHON="python3.12"

# ── Create venv if it doesn't exist ────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment with $PYTHON ..."
  $PYTHON -m venv "$VENV_DIR"
fi

PIP="$VENV_DIR/bin/pip"
STREAMLIT="$VENV_DIR/bin/python -m streamlit"

# ── Install / sync dependencies ─────────────────────────────────────────────
echo "Checking dependencies ..."
$PIP install --quiet \
  streamlit==1.35.0 \
  pandas==2.2.2 \
  "numpy<2" \
  scikit-learn==1.6.1 \
  joblib==1.4.2 \
  langgraph \
  langchain-groq \
  langchain-community \
  langchain-core \
  python-dotenv \
  faiss-cpu \
  sentence-transformers

# ── Pre-build RAG vector store if not already built ─────────────────────────
if [ ! -d "$SCRIPT_DIR/rag_store" ]; then
  echo "Building RAG vector store (first-time setup, ~20s) ..."
  "$VENV_DIR/bin/python" -m rag.setup
fi

# ── Launch Streamlit ────────────────────────────────────────────────────────
PORT="${PORT:-8501}"
echo ""
echo "Starting ValoraAI on http://localhost:$PORT"
echo ""
$STREAMLIT run app/app.py \
  --server.port "$PORT" \
  --server.headless false \
  --server.address localhost
