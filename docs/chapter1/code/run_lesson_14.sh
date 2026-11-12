#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating uv venv..."
  uv venv --python 3.12
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt || true
  # Prefer official CPU wheels if default torch resolve fails/hangs
  if ! python -c "import torch" 2>/dev/null; then
    uv pip install torch --index-url https://download.pytorch.org/whl/cpu
  fi
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo ">>> 05_sparse_vsm_lab (One-Hot / TF-IDF)"
python 05_sparse_vsm_lab.py
echo
echo ">>> 06_ngram_order_lab (词序 / N-gram)"
python 06_ngram_order_lab.py
echo
echo ">>> 07_static_embedding_lab (Word2Vec 静态)"
python 07_static_embedding_lab.py
echo
echo ">>> 08_contextual_bilstm_lab (ELMo 风格上下文)"
python 08_contextual_bilstm_lab.py

echo
echo "Lesson 1.4 chain finished."
