#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv ]]; then
  echo "Creating uv venv..."
  uv venv --python 3.12
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv pip install -r requirements.txt
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

echo ">>> 01_chinese_tokenize"
python 01_chinese_tokenize.py
echo
echo ">>> 02_tfidf_classify"
python 02_tfidf_classify.py
echo
echo ">>> 03_ngram_lm"
python 03_ngram_lm.py
echo
echo ">>> 04_word2vec_skipgram"
python 04_word2vec_skipgram.py

echo
echo "Tip: for the §1.4 representation chain lesson, run: bash run_lesson_14.sh"
echo "All chapter1 demos finished."
