#!/usr/bin/env bash
set -euo pipefail

# Reproduce word-mix retrieval experiments for:
#   EN-ZH, EN-VI, ZH-VI, HI-ID
# This script runs both word-mix retrieval and embedding-mix retrieval while
# sharing one multilingual document index root across all pairs.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"

DATASET=mmarco
REPO=unicamp-dl/mmarco
ENCODER=BAAI/bge-m3
ENC_TAG=${ENCODER##*/}
SIZE=100000
QRELS_SPLIT=validation
DTYPE=fp16
BATCH=128
ENC_BATCH=64
TOPK=100
NEG_PROB=${NEG_PROB:-0.02}

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data/mmarco_dev}"
RUN_TAG="${RUN_TAG:-$(date '+%Y%m%d_%H%M%S')}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs/reproduce_word_mix_${RUN_TAG}}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/reproduce_word_mix_${RUN_TAG}}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/reproduce_word_mix_${RUN_TAG}}"

INDEX_SAVE_ROOT="${INDEX_SAVE_ROOT:-$REPO_ROOT/indexes}"
INDEX_RUN_NAME="${INDEX_RUN_NAME:-idx-${DATASET}-${ENC_TAG}-sub${SIZE}-word-mix}"
DEFAULT_INDEX_ROOT="$INDEX_SAVE_ROOT/$INDEX_RUN_NAME"
INDEX_ROOT="${INDEX_ROOT:-$DEFAULT_INDEX_ROOT}"
if [[ "$INDEX_ROOT" != "$DEFAULT_INDEX_ROOT" ]]; then
  INDEX_SAVE_ROOT=$(dirname "$INDEX_ROOT")
  INDEX_RUN_NAME=$(basename "$INDEX_ROOT")
fi

Q_EN="${Q_EN:-$DATA_ROOT/queries.en.tsv}"
Q_ZH="${Q_ZH:-$DATA_ROOT/queries.zh.tsv}"
Q_VI="${Q_VI:-$DATA_ROOT/queries.vi.tsv}"
Q_HI="${Q_HI:-$DATA_ROOT/queries.hi.tsv}"
Q_ID="${Q_ID:-$DATA_ROOT/queries.id.tsv}"

CM_DIR_EN_ZH="${CM_DIR_EN_ZH:-$DATA_ROOT/queries_cm_5_bands_5-mini}"
CM_DIR_EN_VI="${CM_DIR_EN_VI:-$DATA_ROOT/queries_cm_5_bands_en_vi_5-mini}"
CM_DIR_ZH_VI="${CM_DIR_ZH_VI:-$DATA_ROOT/queries_cm_5_bands_zh_vi_5-mini}"
CM_DIR_HI_ID="${CM_DIR_HI_ID:-$DATA_ROOT/queries_cm_5_bands_hi_id_5-mini}"

COMMON_QIDS_EN_ZH="${COMMON_QIDS_EN_ZH:-$CM_DIR_EN_ZH/qids-common.tsv}"
COMMON_QIDS_EN_VI="${COMMON_QIDS_EN_VI:-$CM_DIR_EN_VI/qids-common.tsv}"
COMMON_QIDS_ZH_VI="${COMMON_QIDS_ZH_VI:-$CM_DIR_ZH_VI/qids-common.tsv}"
COMMON_QIDS_HI_ID="${COMMON_QIDS_HI_ID:-$CM_DIR_HI_ID/qids-common.tsv}"

# Comment out any pair below if you want to run only a subset.
ACTIVE_PAIRS=(
  "en-zh"
  "en-vi"
  "zh-vi"
  "hi-id"
)

# GPU placement. Reuse the same ids if you have fewer GPUs.
GPU_EN=${GPU_EN:-0}
GPU_ZH=${GPU_ZH:-0}
GPU_VI=${GPU_VI:-0}
GPU_HI=${GPU_HI:-0}
GPU_ID=${GPU_ID:-0}
GPU_BI=${GPU_BI:-0}

GPU_INDEX_EN=${GPU_INDEX_EN:-$GPU_EN}
GPU_INDEX_ZH=${GPU_INDEX_ZH:-$GPU_ZH}
GPU_INDEX_VI=${GPU_INDEX_VI:-$GPU_VI}
GPU_INDEX_HI=${GPU_INDEX_HI:-$GPU_HI}
GPU_INDEX_ID=${GPU_INDEX_ID:-$GPU_ID}

DEFAULT_GPU_SLOTS=${DEFAULT_GPU_SLOTS:-1}
GPU0_SLOTS=${GPU0_SLOTS:-2}
GPU1_SLOTS=${GPU1_SLOTS:-2}
GPU2_SLOTS=${GPU2_SLOTS:-1}

ALPHAS=(0 0.1 0.3 0.5 0.7 0.9 1)

FORCE=${FORCE:-0}
SKIP_EVAL=${SKIP_EVAL:-0}
SKIP_INDEX=${SKIP_INDEX:-0}

ONEPASS_DENSE_RUN="$PY_ROOT/onepass_dense_run.py"
ONEPASS_MIX_MONO="$PY_ROOT/onepass_dense_mix_run_custom_lang.py"
ONEPASS_BI_WORD="$PY_ROOT/onepass_bilingual_hub.py"
ONEPASS_BI_MIX="$PY_ROOT/onepass_bilingual_mix_hub_custom_lang.py"
EVAL_SCRIPT="$PY_ROOT/evaluate.py"
ENCODE_SCRIPT="$PY_ROOT/encode_multilingual_corpus.py"

declare -A PAIR_SRC_CODE=(
  ["en-zh"]="en"
  ["en-vi"]="en"
  ["zh-vi"]="zh"
  ["hi-id"]="hi"
)
declare -A PAIR_TGT_CODE=(
  ["en-zh"]="zh"
  ["en-vi"]="vi"
  ["zh-vi"]="vi"
  ["hi-id"]="id"
)
declare -A PAIR_CM_DIR=(
  ["en-zh"]="$CM_DIR_EN_ZH"
  ["en-vi"]="$CM_DIR_EN_VI"
  ["zh-vi"]="$CM_DIR_ZH_VI"
  ["hi-id"]="$CM_DIR_HI_ID"
)
declare -A PAIR_COMMON_QIDS=(
  ["en-zh"]="$COMMON_QIDS_EN_ZH"
  ["en-vi"]="$COMMON_QIDS_EN_VI"
  ["zh-vi"]="$COMMON_QIDS_ZH_VI"
  ["hi-id"]="$COMMON_QIDS_HI_ID"
)
declare -A PAIR_BANDS=()

declare -A GPU_CAPACITY=()
declare -A GPU_SLOT_USAGE=()
declare -A PID_TO_GPU=()
declare -A PID_TO_DESC=()
RUNNING_JOBS=0
FAILED=0

mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_DIR" "$INDEX_SAVE_ROOT"

log() {
  echo "[$(date '+%F %T')] $*"
}

require_file() {
  local path=$1
  if [[ ! -f "$path" ]]; then
    echo "[ERROR] Missing file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path=$1
  if [[ ! -d "$path" ]]; then
    echo "[ERROR] Missing directory: $path" >&2
    exit 1
  fi
}

run_with_log() {
  local log_file=$1
  shift
  mkdir -p "$(dirname "$log_file")"
  "$@" >>"$log_file" 2>&1
}

lang_name_from_code() {
  case "$1" in
    en) echo "english" ;;
    hi) echo "hindi" ;;
    zh) echo "chinese" ;;
    vi) echo "vietnamese" ;;
    id) echo "indonesian" ;;
    *)
      echo "[ERROR] Unsupported language code: $1" >&2
      exit 1
      ;;
  esac
}

query_path_from_code() {
  case "$1" in
    en) echo "$Q_EN" ;;
    hi) echo "$Q_HI" ;;
    zh) echo "$Q_ZH" ;;
    vi) echo "$Q_VI" ;;
    id) echo "$Q_ID" ;;
    *)
      echo "[ERROR] Unsupported query language code: $1" >&2
      exit 1
      ;;
  esac
}

gpu_for_doc_code() {
  case "$1" in
    en) echo "$GPU_EN" ;;
    hi) echo "$GPU_HI" ;;
    zh) echo "$GPU_ZH" ;;
    vi) echo "$GPU_VI" ;;
    id) echo "$GPU_ID" ;;
    *)
      echo "[ERROR] Unsupported document language code: $1" >&2
      exit 1
      ;;
  esac
}

gpu_for_index_code() {
  case "$1" in
    en) echo "$GPU_INDEX_EN" ;;
    hi) echo "$GPU_INDEX_HI" ;;
    zh) echo "$GPU_INDEX_ZH" ;;
    vi) echo "$GPU_INDEX_VI" ;;
    id) echo "$GPU_INDEX_ID" ;;
    *)
      echo "[ERROR] Unsupported index language code: $1" >&2
      exit 1
      ;;
  esac
}

pair_dir_name() {
  local pair=$1
  echo "word_mix_${pair//-/_}"
}

pair_run_root() {
  local pair=$1
  echo "$RUN_ROOT/$(pair_dir_name "$pair")"
}

pair_result_root() {
  local pair=$1
  echo "$RESULT_ROOT/$(pair_dir_name "$pair")"
}

pair_log_root() {
  local pair=$1
  echo "$LOG_DIR/$(pair_dir_name "$pair")"
}

pair_bands_array() {
  local pair=$1
  local bands_string=${PAIR_BANDS[$pair]:-}
  if [[ -z "$bands_string" ]]; then
    echo "[ERROR] No bands registered for pair: $pair" >&2
    exit 1
  fi
  echo "$bands_string"
}

discover_pair_bands() {
  local pair=$1
  local cm_dir=${PAIR_CM_DIR[$pair]:-}
  local file band
  local -a bands=()

  require_dir "$cm_dir"
  shopt -s nullglob
  for file in "$cm_dir"/queries-cm*.tsv; do
    band=$(basename "$file")
    band=${band#queries-cm}
    band=${band%.tsv}
    bands+=("$band")
  done
  shopt -u nullglob

  if [[ ${#bands[@]} -eq 0 ]]; then
    echo "[ERROR] No queries-cm*.tsv files found in $cm_dir" >&2
    exit 1
  fi

  mapfile -t bands < <(printf '%s\n' "${bands[@]}" | sort -V)
  PAIR_BANDS["$pair"]="${bands[*]}"
  log "Using code-mix bands for ${pair}: ${bands[*]}"
}

index_ready() {
  local lang_name=$1
  [[ -f "$INDEX_ROOT/$lang_name/index.faiss" &&
     -f "$INDEX_ROOT/$lang_name/docid_map.tsv" &&
     -f "$INDEX_ROOT/$lang_name/docids.txt" ]]
}

register_gpu() {
  local gpu=$1
  if [[ -n "${GPU_CAPACITY[$gpu]:-}" ]]; then
    return
  fi
  local cap_var="GPU${gpu}_SLOTS"
  local cap=${!cap_var:-$DEFAULT_GPU_SLOTS}
  if (( cap <= 0 )); then
    echo "[ERROR] ${cap_var} resolves to ${cap}; GPU ${gpu} cannot accept jobs." >&2
    exit 1
  fi
  GPU_CAPACITY["$gpu"]="$cap"
  GPU_SLOT_USAGE["$gpu"]=0
}

reap_finished_jobs() {
  local pid status gpu desc
  for pid in "${!PID_TO_GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      status=0
      wait "$pid" || status=$?
      gpu=${PID_TO_GPU[$pid]}
      desc=${PID_TO_DESC[$pid]}
      unset PID_TO_GPU["$pid"] PID_TO_DESC["$pid"]
      if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
        GPU_SLOT_USAGE["$gpu"]=$((GPU_SLOT_USAGE[$gpu] - 1))
      fi
      RUNNING_JOBS=$((RUNNING_JOBS - 1))
      if [[ $status -ne 0 ]]; then
        FAILED=1
        log "Job failed: ${desc} (status ${status})"
      else
        log "Job finished: ${desc}"
      fi
    fi
  done
}

wait_for_gpu_slot() {
  local gpu=$1
  while true; do
    reap_finished_jobs
    if (( GPU_SLOT_USAGE[$gpu] < GPU_CAPACITY[$gpu] )); then
      return
    fi
    sleep 5
  done
}

start_job() {
  local gpu=$1
  local desc=$2
  shift 2
  register_gpu "$gpu"
  wait_for_gpu_slot "$gpu"
  log "Starting job: ${desc} (GPU ${gpu})"
  (
    set -euo pipefail
    "$@"
  ) &
  local pid=$!
  PID_TO_GPU[$pid]=$gpu
  PID_TO_DESC[$pid]=$desc
  GPU_SLOT_USAGE["$gpu"]=$((GPU_SLOT_USAGE[$gpu] + 1))
  RUNNING_JOBS=$((RUNNING_JOBS + 1))
}

wait_for_all_jobs() {
  while (( RUNNING_JOBS > 0 )); do
    reap_finished_jobs
    sleep 5
  done
}

encode_index() {
  local lang_code=$1
  local gpu=$2
  local lang_name
  lang_name=$(lang_name_from_code "$lang_code")
  local log_file="$LOG_DIR/index_${lang_name}_${RUN_TAG}.log"
  log "Encoding ${lang_name} corpus index into $INDEX_ROOT (GPU ${gpu})"
  run_with_log "$log_file" \
    python "$ENCODE_SCRIPT" \
      --repo "$REPO" \
      --split collection \
      --encoder "$ENCODER" \
      --device "cuda:${gpu}" \
      --batch "$BATCH" \
      --enc_batch "$ENC_BATCH" \
      --dtype "$DTYPE" \
      --gpu_faiss \
      --faiss_gpu_id "${gpu}" \
      --langs "${lang_name}" \
      --subset_neg_cap "$SIZE" \
      --neg_prob "$NEG_PROB" \
      --qrels_repo BeIR/msmarco-qrels \
      --qrels_split "$QRELS_SPLIT" \
      --qrels_docid corpus-id \
      --trust_remote \
      --save_root "$INDEX_SAVE_ROOT" \
      --run_name "$INDEX_RUN_NAME"
}

ensure_indexes() {
  local -A needed_codes=()
  local pair code gpu

  if [[ "$SKIP_INDEX" -eq 1 ]]; then
    log "SKIP_INDEX=1, skipping document encoding and assuming indexes already exist at $INDEX_ROOT"
    return
  fi

  for pair in "${ACTIVE_PAIRS[@]}"; do
    needed_codes["${PAIR_SRC_CODE[$pair]}"]=1
    needed_codes["${PAIR_TGT_CODE[$pair]}"]=1
  done

  for code in "${!needed_codes[@]}"; do
    if [[ "$FORCE" -eq 1 ]] || ! index_ready "$(lang_name_from_code "$code")"; then
      gpu=$(gpu_for_index_code "$code")
      start_job "$gpu" "encode $(lang_name_from_code "$code")" encode_index "$code" "$gpu"
    else
      log "Index already present for $(lang_name_from_code "$code") under $INDEX_ROOT; skipping encoding."
    fi
  done

  wait_for_all_jobs
  if [[ "$FAILED" -ne 0 ]]; then
    echo "[ERROR] One or more index jobs failed. Check logs under $LOG_DIR." >&2
    exit 1
  fi
}

run_mono_wordmix() {
  local pair=$1
  local doc_code=$2
  local gpu=$3

  local doc_lang
  doc_lang=$(lang_name_from_code "$doc_code")
  local cm_dir=${PAIR_CM_DIR[$pair]}
  local common_qids=${PAIR_COMMON_QIDS[$pair]}
  local pair_root
  pair_root=$(pair_run_root "$pair")
  local pair_result
  pair_result=$(pair_result_root "$pair")
  local pair_logs
  pair_logs=$(pair_log_root "$pair")
  local run_dir="$pair_root/${DATASET}-${SIZE}-${doc_lang}-${pair}-word-mix-${ENC_TAG}"
  local result_dir="$pair_result/${DATASET}-${SIZE}-${doc_lang}-${pair}-word-mix-${ENC_TAG}"
  local docids_out="$run_dir/docids.txt"
  local log_file="$pair_logs/mono-wordmix-${doc_lang}-$(date '+%Y%m%d_%H%M%S').log"
  local bands_string
  local -a bands
  local band

  bands_string=$(pair_bands_array "$pair")
  read -r -a bands <<< "$bands_string"
  mkdir -p "$run_dir" "$result_dir" "$pair_logs"

  log "Monolingual word-mix for ${pair} on ${doc_lang} docs (GPU ${gpu})"

  if [[ ! -f "$run_dir/cm${bands[0]}.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_DENSE_RUN" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --q_config "queries-${doc_lang}" \
        --q_split dev \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --run_out "$run_dir" \
        --docids_out "$docids_out" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --q_directory "$cm_dir" \
        --q_glob "queries-cm[0-9]*.tsv" \
        --max_docs "$SIZE"
  else
    log "Skipping existing monolingual word-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for band in "${bands[@]}"; do
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm${band}.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_out" \
          --filter_qids "$common_qids"
    done
  fi
}

run_mono_vecmix() {
  local pair=$1
  local doc_code=$2
  local gpu=$3

  local src_code=${PAIR_SRC_CODE[$pair]}
  local tgt_code=${PAIR_TGT_CODE[$pair]}
  local doc_lang
  doc_lang=$(lang_name_from_code "$doc_code")
  local common_qids=${PAIR_COMMON_QIDS[$pair]}
  local pair_root
  pair_root=$(pair_run_root "$pair")
  local pair_result
  pair_result=$(pair_result_root "$pair")
  local pair_logs
  pair_logs=$(pair_log_root "$pair")
  local run_dir="$pair_root/${DATASET}-${SIZE}-${doc_lang}-${pair}-word-mix-${ENC_TAG}/vector_mix"
  local result_dir="$pair_result/${DATASET}-${SIZE}-${doc_lang}-${pair}-word-mix-${ENC_TAG}/vector_mix"
  local docids_path="$run_dir/docids-${pair}.txt"
  local log_file="$pair_logs/mono-vecmix-${doc_lang}-$(date '+%Y%m%d_%H%M%S').log"
  local alpha

  mkdir -p "$run_dir" "$result_dir" "$pair_logs"
  log "Monolingual embedding-mix for ${pair} on ${doc_lang} docs (GPU ${gpu})"

  if [[ ! -f "$run_dir/cm-alpha-0.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_MIX_MONO" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --run_out "$run_dir" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --query_tsv "${src_code}=$(query_path_from_code "$src_code")" \
        --query_tsv "${tgt_code}=$(query_path_from_code "$tgt_code")" \
        --cm_alphas "0,0.1,0.3,0.5,0.7,0.9,1"
  else
    log "Skipping existing monolingual embedding-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for alpha in "${ALPHAS[@]}"; do
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm-alpha-${alpha}.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$common_qids"
    done
  fi
}

run_bi_wordmix() {
  local pair=$1
  local gpu=$2

  local src_code=${PAIR_SRC_CODE[$pair]}
  local tgt_code=${PAIR_TGT_CODE[$pair]}
  local src_lang
  src_lang=$(lang_name_from_code "$src_code")
  local tgt_lang
  tgt_lang=$(lang_name_from_code "$tgt_code")
  local cm_dir=${PAIR_CM_DIR[$pair]}
  local common_qids=${PAIR_COMMON_QIDS[$pair]}
  local pair_root
  pair_root=$(pair_run_root "$pair")
  local pair_result
  pair_result=$(pair_result_root "$pair")
  local pair_logs
  pair_logs=$(pair_log_root "$pair")
  local run_dir="$pair_root/${DATASET}-${SIZE}-bilingual-${pair}-word-mix-${ENC_TAG}"
  local result_dir="$pair_result/${DATASET}-${SIZE}-bilingual-${pair}-word-mix-${ENC_TAG}"
  local docids_path="$run_dir/docids.txt"
  local log_file="$pair_logs/bilingual-wordmix-$(date '+%Y%m%d_%H%M%S').log"
  local bands_string
  local -a bands
  local band

  bands_string=$(pair_bands_array "$pair")
  read -r -a bands <<< "$bands_string"
  mkdir -p "$run_dir" "$result_dir" "$pair_logs"

  log "Bilingual word-mix for ${pair} on ${src_lang}+${tgt_lang} docs (GPU ${gpu})"

  if [[ ! -f "$run_dir/cm${bands[0]}_base.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_BI_WORD" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --enc_batch "$ENC_BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --langs "${src_lang},${tgt_lang}" \
        --q_directory "$cm_dir" \
        --outdir "$run_dir"
  else
    log "Skipping existing bilingual word-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for band in "${bands[@]}"; do
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm${band}_base.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$common_qids"
    done
  fi
}

run_bi_vecmix() {
  local pair=$1
  local gpu=$2

  local src_code=${PAIR_SRC_CODE[$pair]}
  local tgt_code=${PAIR_TGT_CODE[$pair]}
  local src_lang
  src_lang=$(lang_name_from_code "$src_code")
  local tgt_lang
  tgt_lang=$(lang_name_from_code "$tgt_code")
  local common_qids=${PAIR_COMMON_QIDS[$pair]}
  local pair_root
  pair_root=$(pair_run_root "$pair")
  local pair_result
  pair_result=$(pair_result_root "$pair")
  local pair_logs
  pair_logs=$(pair_log_root "$pair")
  local run_dir="$pair_root/${DATASET}-${SIZE}-bilingual-${pair}-word-mix-${ENC_TAG}/vector_mix"
  local result_dir="$pair_result/${DATASET}-${SIZE}-bilingual-${pair}-word-mix-${ENC_TAG}/vector_mix"
  local docids_path="$run_dir/docids-${pair}.txt"
  local log_file="$pair_logs/bilingual-vecmix-$(date '+%Y%m%d_%H%M%S').log"
  local alpha

  mkdir -p "$run_dir" "$result_dir" "$pair_logs"
  log "Bilingual embedding-mix for ${pair} on ${src_lang}+${tgt_lang} docs (GPU ${gpu})"

  if [[ ! -f "$run_dir/cm-alpha-0.trec" || "$FORCE" -eq 1 ]]; then
    run_with_log "$log_file" \
      python "$ONEPASS_BI_MIX" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --langs "${src_lang},${tgt_lang}" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$ENCODER" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --index_root "$INDEX_ROOT" \
        --trust_remote \
        --gpu_faiss --faiss_gpu_id "${gpu}" \
        --batch "$BATCH" \
        --dtype "$DTYPE" \
        --max_docs "$SIZE" \
        --query_tsv "${src_code}=$(query_path_from_code "$src_code")" \
        --query_tsv "${tgt_code}=$(query_path_from_code "$tgt_code")" \
        --outdir "$run_dir" \
        --cm_alphas "0,0.1,0.3,0.5,0.7,0.9,1"
  else
    log "Skipping existing bilingual embedding-mix runs in $run_dir"
  fi

  if [[ "$SKIP_EVAL" -ne 1 ]]; then
    for alpha in "${ALPHAS[@]}"; do
      run_with_log "$log_file" \
        python "$EVAL_SCRIPT" \
          --dataset "$DATASET" \
          --run "$run_dir/cm-alpha-${alpha}.trec" \
          --qrels_repo BeIR/msmarco-qrels \
          --qrels_split "$QRELS_SPLIT" \
          --outdir "$result_dir" \
          --trust_remote \
          --filter_docids "$docids_path" \
          --filter_qids "$common_qids"
    done
  fi
}

require_file "$ONEPASS_DENSE_RUN"
require_file "$ONEPASS_MIX_MONO"
require_file "$ONEPASS_BI_WORD"
require_file "$ONEPASS_BI_MIX"
require_file "$EVAL_SCRIPT"
require_file "$ENCODE_SCRIPT"

declare -A REQUIRED_CODES=()
for pair in "${ACTIVE_PAIRS[@]}"; do
  if [[ -z "${PAIR_SRC_CODE[$pair]:-}" || -z "${PAIR_TGT_CODE[$pair]:-}" ]]; then
    echo "[ERROR] Unsupported pair in ACTIVE_PAIRS: $pair" >&2
    exit 1
  fi
  REQUIRED_CODES["${PAIR_SRC_CODE[$pair]}"]=1
  REQUIRED_CODES["${PAIR_TGT_CODE[$pair]}"]=1
  require_dir "${PAIR_CM_DIR[$pair]}"
  require_file "${PAIR_COMMON_QIDS[$pair]}"
  discover_pair_bands "$pair"
done

for code in "${!REQUIRED_CODES[@]}"; do
  require_file "$(query_path_from_code "$code")"
done

for gpu in \
  "$GPU_EN" "$GPU_ZH" "$GPU_VI" "$GPU_HI" "$GPU_ID" "$GPU_BI" \
  "$GPU_INDEX_EN" "$GPU_INDEX_ZH" "$GPU_INDEX_VI" "$GPU_INDEX_HI" "$GPU_INDEX_ID"
do
  register_gpu "$gpu"
done

log "=== word-mix reproduction start ==="
log "Shared index root: $INDEX_ROOT"
ensure_indexes

FAILED=0
RUNNING_JOBS=0
PID_TO_GPU=()
PID_TO_DESC=()
for gpu in "${!GPU_SLOT_USAGE[@]}"; do
  GPU_SLOT_USAGE["$gpu"]=0
done

for pair in "${ACTIVE_PAIRS[@]}"; do
  src_code=${PAIR_SRC_CODE[$pair]}
  tgt_code=${PAIR_TGT_CODE[$pair]}
  start_job "$(gpu_for_doc_code "$src_code")" "${pair} mono-$(lang_name_from_code "$src_code") word-mix" \
    run_mono_wordmix "$pair" "$src_code" "$(gpu_for_doc_code "$src_code")"
  start_job "$(gpu_for_doc_code "$src_code")" "${pair} mono-$(lang_name_from_code "$src_code") vec-mix" \
    run_mono_vecmix "$pair" "$src_code" "$(gpu_for_doc_code "$src_code")"
  start_job "$(gpu_for_doc_code "$tgt_code")" "${pair} mono-$(lang_name_from_code "$tgt_code") word-mix" \
    run_mono_wordmix "$pair" "$tgt_code" "$(gpu_for_doc_code "$tgt_code")"
  start_job "$(gpu_for_doc_code "$tgt_code")" "${pair} mono-$(lang_name_from_code "$tgt_code") vec-mix" \
    run_mono_vecmix "$pair" "$tgt_code" "$(gpu_for_doc_code "$tgt_code")"
  start_job "$GPU_BI" "${pair} bilingual word-mix" run_bi_wordmix "$pair" "$GPU_BI"
  start_job "$GPU_BI" "${pair} bilingual vec-mix" run_bi_vecmix "$pair" "$GPU_BI"
done

wait_for_all_jobs
if [[ "$FAILED" -ne 0 ]]; then
  echo "[ERROR] One or more jobs failed. Check logs under $LOG_DIR." >&2
  exit 1
fi

log "=== word-mix reproduction done ==="
