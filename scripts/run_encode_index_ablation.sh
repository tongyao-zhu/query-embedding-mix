#!/usr/bin/env bash
set -euo pipefail

# Encode + index the corpus for the ablation models, mirroring run_encode_index_groups.sh
# but targeting only the languages used in the ablation plan.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET=${DATASET:-mmarco}
REPO=${REPO:-unicamp-dl/mmarco}
SPLIT=${SPLIT:-collection}
INDEX_ROOT_BASE=${INDEX_ROOT_BASE:-$REPO_ROOT/indexes}
LOG_DIR="${LOG_DIR:-$REPO_ROOT/index_logs_ablation}"
mkdir -p "$LOG_DIR"

# Devices / scheduler
GPUS=(${GPUS:-0})
DEFAULT_GPU_SLOTS=${DEFAULT_GPU_SLOTS:-1}
declare -A GPU_CAPACITY=(
    [0]=${GPU0_SLOTS:-1}
    [1]=${GPU1_SLOTS:-1}
)
declare -A GPU_SLOT_USAGE=()
for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_CAPACITY[$gpu]:-}" ]]; then
        GPU_CAPACITY[$gpu]=$DEFAULT_GPU_SLOTS
    fi
    GPU_SLOT_USAGE[$gpu]=0
done
SLEEP_INTERVAL=${SLEEP_INTERVAL:-120}

# Languages needed (doc side for all jobs)
declare -A LANG_CONFIG_MAP=(
    [en]=english
    [ar]=arabic
    [zh]=chinese
    [de]=german
    [nl]=dutch
    [ru]=russian
)
LANG_CODES=(en ar zh de nl ru)

lang_cfg() {
    local code=$1
    local cfg=${LANG_CONFIG_MAP[$code]:-}
    if [[ -z "$cfg" ]]; then
        echo "[ERROR] Unsupported language code: $code" >&2
        exit 1
    fi
    echo "$cfg"
}

# Encoder definitions
declare -A ENCODER_NAME=(
    [me5-large-instruct]="intfloat/multilingual-e5-large-instruct"
    [gte-multilingual-base]="Alibaba-NLP/gte-multilingual-base"
    [jina-embedding-v3]="jinaai/jina-embeddings-v3"
    [qwen3-embedding-0.6B]="Qwen/Qwen3-Embedding-0.6B"
    [qwen3-embedding-4B]="Qwen/Qwen3-Embedding-4B"
    [qwen3-embedding-8B]="Qwen/Qwen3-Embedding-8B"
)

# Defaults (override via env ENC_BATCH_<TAG>, BATCH_<TAG>, DEVICE_<TAG>)
declare -A ENC_BATCH_DEFAULT=(
    [me5-large-instruct]=128
    [gte-multilingual-base]=128
    [jina-embedding-v3]=64
    [qwen3-embedding-0.6B]=64
    [qwen3-embedding-4B]=32
    [qwen3-embedding-8B]=16
)
declare -A BATCH_DEFAULT=(
    [me5-large-instruct]=32768
    [gte-multilingual-base]=32768
    [jina-embedding-v3]=32768
    [qwen3-embedding-0.6B]=32768
    [qwen3-embedding-4B]=8192
    [qwen3-embedding-8B]=32768
)
declare -A DEVICE_DEFAULT=(
    [me5-large-instruct]="cuda:0"
    [gte-multilingual-base]="cuda:0"
    [jina-embedding-v3]="cuda:0"
    [qwen3-embedding-0.6B]="cuda:0"
    [qwen3-embedding-4B]="cuda:1"
    [qwen3-embedding-8B]="cuda:1"
)

# Skip flags (set to 1 to skip a model family)
SKIP_E5=${SKIP_E5:-0}
SKIP_GTE=${SKIP_GTE:-0}
SKIP_JINA=${SKIP_JINA:-0}
SKIP_QWEN06=${SKIP_QWEN06:-0}
SKIP_QWEN4=${SKIP_QWEN4:-0}
SKIP_QWEN8=${SKIP_QWEN8:-0}

# Subset / FAISS tuning
SUBSET_NEG_CAP=${SUBSET_NEG_CAP:-100000}   # empty = full corpus per lang
NEG_PROB=${NEG_PROB:-1.0}
GPU_FAISS=${GPU_FAISS:-0}
FAISS_GPU_ID=${FAISS_GPU_ID:-0}
DTYPE=${DTYPE:-fp16}
TRUST_REMOTE=${TRUST_REMOTE:-1}
CLEAR_MODEL_CACHE=${CLEAR_MODEL_CACHE:-1}

# Per-model Python deps (space-separated). Script will skip a job if deps missing.
declare -A MODEL_PY_REQS=(
    [jina-embedding-v3]="einops"
)

# Extra dynamic-module repos to clear per model (space-separated).
declare -A MODEL_EXTRA_MODULES=(
    [jina-embedding-v3]="jinaai/xlm-roberta-flash-implementation"
)

sanitize_var_key() {
    echo "$1" | tr '/.-' '___' | tr '[:lower:]' '[:upper:]'
}

get_override() {
    local prefix=$1
    local key=$2
    local def=$3
    local var="${prefix}_$(sanitize_var_key "$key")"
    if [[ -n "${!var:-}" ]]; then
        echo "${!var}"
    else
        echo "$def"
    fi
}

index_lang_ready() {
    local root=$1
    local lang_cfg=$2
    [[ -f "${root}/${lang_cfg}/index.faiss" && -f "${root}/${lang_cfg}/docid_map.tsv" && -f "${root}/${lang_cfg}/docids.txt" ]]
}

cache_base_dir() {
    local base="${HF_HOME:-${HUGGINGFACE_HUB_CACHE:-$HOME/.cache/huggingface}}"
    echo "$base"
}

maybe_clear_model_cache() {
    local encoder=$1
    local key=$2
    if (( ! CLEAR_MODEL_CACHE )); then
        return
    fi
    local base
    base=$(cache_base_dir)
    local -a repos=("${encoder}")
    if [[ -n "${MODEL_EXTRA_MODULES[$key]:-}" ]]; then
        # shellcheck disable=SC2206
        repos+=(${MODEL_EXTRA_MODULES[$key]})
    fi
    for repo_dir in "${repos[@]}"; do
        local path="${base}/modules/transformers_modules/${repo_dir}"
        if [[ -d "$path" ]]; then
            echo "[INFO] Clearing cached modules for ${repo_dir} at ${path}" >&2
            rm -rf -- "$path"
        fi
    done
}

wait_for_gpu_slot() {
    local preferred=${1:-}
    while true; do
        for pid in "${!PID_TO_GPU[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                local status=0
                wait "$pid" || status=$?
                local gpu=${PID_TO_GPU[$pid]}
                local desc=${PID_TO_DESC[$pid]}
                local log=${PID_TO_LOG[$pid]}
                unset PID_TO_GPU["$pid"] PID_TO_DESC["$pid"] PID_TO_LOG["$pid"]
                if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
                    GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] - 1))
                fi
                if [[ $status -ne 0 ]]; then
                    echo "[ERROR] Job '${desc}' failed with status ${status}. See ${log}" >&2
                    exit $status
                fi
                echo "[$(date '+%F %T')] Completed ${desc} (log: ${log})" >&2
            fi
        done

        if [[ -n "$preferred" ]]; then
            local cap=${GPU_CAPACITY[$preferred]:-$DEFAULT_GPU_SLOTS}
            if (( GPU_SLOT_USAGE[$preferred] < cap )); then
                echo "$preferred"
                return
            fi
        else
            for gpu in "${GPUS[@]}"; do
                local cap=${GPU_CAPACITY[$gpu]:-$DEFAULT_GPU_SLOTS}
                if (( GPU_SLOT_USAGE[$gpu] < cap )); then
                    echo "$gpu"
                    return
                fi
            done
        fi
        sleep "$SLEEP_INTERVAL"
    done
}

start_job() {
    local gpu=$1
    local desc=$2
    local log_file=$3
    shift 3
    (
        set -euo pipefail
        "$@"
    ) &> "$log_file" &
    local pid=$!
    PID_TO_GPU[$pid]=$gpu
    PID_TO_DESC[$pid]=$desc
    PID_TO_LOG[$pid]=$log_file
    GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] + 1))
    echo "[$(date '+%F %T')] Launched ${desc} on GPU ${gpu} (log: ${log_file})" >&2
}

declare -A PID_TO_GPU=()
declare -A PID_TO_DESC=()
declare -A PID_TO_LOG=()

build_job() {
    local key=$1
    local encoder=${ENCODER_NAME[$key]}
    local enc_tag=${encoder##*/}
    local index_root="${INDEX_ROOT_BASE}/idx-${DATASET}-${enc_tag}-sub${SUBSET_NEG_CAP:-full}"

    local -a cfgs=()
    local -a missing_cfgs=()
    for code in "${LANG_CODES[@]}"; do
        local cfg
        cfg=$(lang_cfg "$code")
        cfgs+=("$cfg")
        if ! index_lang_ready "$index_root" "$cfg"; then
            missing_cfgs+=("$cfg")
        fi
    done
    if (( ${#missing_cfgs[@]} == 0 )); then
        echo "[INFO] Indexes already present for ${key} at ${index_root}; skipping." >&2
        return
    fi

    local enc_batch
    enc_batch=$(get_override ENC_BATCH "$key" "${ENC_BATCH_DEFAULT[$key]}")
    local batch
    batch=$(get_override BATCH "$key" "${BATCH_DEFAULT[$key]}")
    local device
    device=$(get_override DEVICE "$key" "${DEVICE_DEFAULT[$key]}")
    local pref_gpu=""
    if [[ "$device" == cuda:* ]]; then
        pref_gpu=${device#cuda:}
    fi

    local langs_csv
    langs_csv=$(IFS=','; echo "${missing_cfgs[*]}")
    local run_name="idx-${DATASET}-${enc_tag}-sub${SUBSET_NEG_CAP:-full}"

    local deps=${MODEL_PY_REQS[$key]:-}
    if [[ -n "$deps" ]]; then
        local missing
        missing=$(REQS="$deps" "$PYTHON_BIN" - <<'PY'
import importlib.util, os
reqs = os.environ.get("REQS", "").split()
missing = [r for r in reqs if importlib.util.find_spec(r) is None]
print(",".join(missing))
PY
)
        if [[ -n "$missing" ]]; then
            echo "[ERROR] Missing Python deps for ${key}: ${missing}. Install them (e.g., pip install ${missing//,/ }) and rerun. Skipping job." >&2
            return
        fi
    fi

    maybe_clear_model_cache "$encoder" "$key"

    local log_file="${LOG_DIR}/encode-${key}-$(date '+%Y%m%d_%H%M%S').log"
    local gpu
    if [[ -n "$pref_gpu" ]]; then
        gpu=$(wait_for_gpu_slot "$pref_gpu")
    else
        gpu=$(wait_for_gpu_slot)
    fi

    # If device not specified, bind it to the scheduled GPU.
    local device_for_cmd="$device"
    if [[ -z "$device_for_cmd" && -n "$gpu" ]]; then
        device_for_cmd="cuda:${gpu}"
    fi

    local -a cmd=(
        "$PYTHON_BIN" "${PY_ROOT}/encode_multilingual_corpus.py"
        --repo "$REPO"
        --encoder "$encoder"
        --split "$SPLIT"
        --langs "$langs_csv"
        --run_name "$run_name"
        --save_root "$INDEX_ROOT_BASE"
        --batch "$batch"
        --enc_batch "$enc_batch"
        --device "$device_for_cmd"
        --neg_prob "$NEG_PROB"
        --dtype "$DTYPE"
    )
    if (( TRUST_REMOTE )); then
        cmd+=(--trust_remote)
    fi
    if [[ -n "$SUBSET_NEG_CAP" ]]; then
        cmd+=(--subset_neg_cap "$SUBSET_NEG_CAP")
    fi
    if (( GPU_FAISS )); then
        cmd+=(--gpu_faiss --faiss_gpu_id "$FAISS_GPU_ID")
    fi

    start_job "$gpu" "encode-${key}" "$log_file" "${cmd[@]}"
}

# Build job list respecting skip flags
(( ! SKIP_QWEN4 )) && build_job "qwen3-embedding-4B"
(( ! SKIP_JINA )) && build_job "jina-embedding-v3"
(( ! SKIP_QWEN06 )) && build_job "qwen3-embedding-0.6B"
(( ! SKIP_QWEN8 )) && build_job "qwen3-embedding-8B"
(( ! SKIP_E5 )) && build_job "me5-large-instruct"
(( ! SKIP_GTE )) && build_job "gte-multilingual-base"

while (( ${#PID_TO_GPU[@]} > 0 )); do
    wait_for_gpu_slot >/dev/null
done

echo "[$(date '+%F %T')] All encoding jobs completed."
