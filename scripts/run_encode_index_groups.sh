#!/usr/bin/env bash
set -euo pipefail

# Launch encode_multilingual_corpus.py for language subsets in parallel.
# All groups share the same run directory; each language now writes to its own subfolder/index.
# Adjust constants below as needed.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"
PYTHON_BIN=${PYTHON_BIN:-python}

REPO="unicamp-dl/mmarco"
ENCODER="BAAI/bge-m3"
SPLIT="collection"
SUBSET_CAP=8841823
RUN_NAME_BASE="idx-mmarco-bge-m3-sub${SUBSET_CAP}"
SAVE_ROOT="${INDEX_ROOT_BASE:-$REPO_ROOT/indexes}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/index_logs}"
mkdir -p "${SAVE_ROOT}"
mkdir -p "${LOG_DIR}"

# Uncomment to enable GPU FAISS during indexing
# GPU_FAISS_FLAG="--gpu_faiss --faiss_gpu_id 1"

# Define language groups using dataset config suffixes (comma-separated).
# For mMARCO HF repo these are: english, french, german, italian, spanish,
# portuguese, dutch, russian, japanese, chinese, arabic, hindi,
# indonesian, vietnamese.
GROUP1="english,german,french,italian"
GROUP2="spanish,portuguese,dutch"
GROUP3="russian,japanese,chinese"
GROUP4="arabic,hindi,indonesian,vietnamese"
ALL_LANGS="${GROUP1},${GROUP2},${GROUP3},${GROUP4}"

launch_group() {
  local group_name=$1
  local langs=$2
  local run_name="${RUN_NAME_BASE}"
  echo "[INFO] Launching ${group_name} → ${langs}"
  ${PYTHON_BIN} "${PY_ROOT}/encode_multilingual_corpus.py" \
    --repo "${REPO}" \
    --encoder "${ENCODER}" \
    --split "${SPLIT}" \
    --langs "${langs}" \
    --run_name "${run_name}" \
    --save_root "${SAVE_ROOT}" \
    --batch 32768 \
    --enc_batch 128 \
    --device "cuda:0" \
    --neg_prob 1.0 \
    --verbosity 1 >"${LOG_DIR}/logs_encode_${group_name}.log" 2>&1 &
}

# Previous parallel launcher (kept for reference)
launch_group "g1" "${GROUP1}"
launch_group "g2" "${GROUP2}"
launch_group "g3" "${GROUP3}"
launch_group "g4" "${GROUP4}"

wait
echo "[INFO] All groups finished."

# echo "[INFO] Running sequential subset encoding for all languages (cap=${SUBSET_CAP})"
# ${PYTHON_BIN} "${SCRIPT_DIR}/encode_multilingual_corpus.py" \
#   --repo "${REPO}" \
#   --encoder "${ENCODER}" \
#   --split "${SPLIT}" \
#   --langs "${ALL_LANGS}" \
#   --run_name "${RUN_NAME_BASE}" \
#   --save_root "${SAVE_ROOT}" \
#   --batch 32768 \
#   --enc_batch 128 \
#   --device "cuda:1" \
#   --neg_prob 1.0 \
#   --subset_neg_cap "${SUBSET_CAP}" \
#   --verbosity 1 >"${LOG_DIR}/logs_encode_all_sub${SUBSET_CAP}.log" 2>&1

# echo "[INFO] Sequential encoding complete: ${RUN_NAME_BASE}"
