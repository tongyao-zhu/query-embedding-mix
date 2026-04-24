#!/usr/bin/env bash
set -euo pipefail

# Ablation runner for vector experiments across multiple embedding models.
# Reuses the scheduling + eval flow from run_all_vector_pairs.sh but pins the
# job matrix to the requested ablation plan.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET=mmarco
REPO=unicamp-dl/mmarco
SIZE=${SIZE:-100000}
QRELS_SPLIT=validation
CM_ALPHAS="0,0.1,0.3,0.5,0.7,0.9,1"
BANDS=(0 0.1 0.3 0.5 0.7 0.9 1)

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/mmarco_full}"
QUERY_DIR="${QUERY_DIR:-$DATA_ROOT/mmarco_dev}"
COMMON_QIDS="${COMMON_QIDS:-$REPO_ROOT/configs/qid_lists/qids-common.tsv}"

BILINGUAL_SCRIPT="${PY_ROOT}/onepass_bilingual_mix_hub_custom_lang.py"
MONO_SCRIPT="${PY_ROOT}/onepass_dense_mix_run_custom_lang.py"
EVAL_SCRIPT="${PY_ROOT}/evaluate.py"
CACHE_SCRIPT="${PY_ROOT}/cache_queries_for_mix.py"
QRELS_CACHE="${QRELS_CACHE:-$DATA_ROOT/qrels_cache}"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/ablation}"
mkdir -p "$LOG_DIR" "$RESULT_ROOT" "$RUN_ROOT"

INDEX_ROOT_OVERRIDE=${INDEX_ROOT_OVERRIDE:-${INDEX_ROOT:-}}
QUERY_CACHE_ROOT_OVERRIDE=${QUERY_CACHE_ROOT_OVERRIDE:-${QUERY_CACHE_ROOT:-}}
PREENCODE_QUERIES=${PREENCODE_QUERIES:-1}
PREENC_QUERY_DEVICE=${PREENC_QUERY_DEVICE:-cuda:1}
PREENC_QUERY_BATCH=${PREENC_QUERY_BATCH:-16}
PREENC_QUERY_TRUST_REMOTE=${PREENC_QUERY_TRUST_REMOTE:-1}
INDEX_SUBSET_TAG=${INDEX_SUBSET_TAG:-sub100000}

GPUS=(${GPUS:-0})
SLEEP_INTERVAL=${SLEEP_INTERVAL:-60}       # seconds between scheduler polls
BETWEEN_LAUNCH_SLEEP=${BETWEEN_LAUNCH_SLEEP:-5}
DEFAULT_GPU_SLOTS=${DEFAULT_GPU_SLOTS:-1}

# Allow a simple override for per-GPU capacity; fall back to run_all defaults.
declare -A GPU_CAPACITY=(
    [0]=${GPU0_SLOTS:-3}
    [1]=${GPU1_SLOTS:-0}
)
declare -A GPU_SLOT_USAGE=()
for gpu in "${GPUS[@]}"; do
    if [[ -z "${GPU_CAPACITY[$gpu]:-}" ]]; then
        GPU_CAPACITY[$gpu]=$DEFAULT_GPU_SLOTS
    fi
    GPU_SLOT_USAGE[$gpu]=0
done

declare -A PID_TO_GPU=()
declare -A PID_TO_DESC=()
declare -A PID_TO_LOG=()
RUNNING_JOBS=0

declare -A LANG_NAME_MAP=(
    [ar]=arabic
    [de]=german
    [en]=english
    [es]=spanish
    [fr]=french
    [hi]=hindi
    [id]=indonesian
    [it]=italian
    [ja]=japanese
    [nl]=dutch
    [pt]=portuguese
    [ru]=russian
    [vi]=vietnamese
    [zh]=chinese
)

declare -A QUERY_FILE_MAP=()
for code in "${!LANG_NAME_MAP[@]}"; do
    QUERY_FILE_MAP[$code]="${QUERY_DIR}/queries.${code}.tsv"
done

lang_config_from_code() {
    local code=$1
    local cfg=${LANG_NAME_MAP[$code]:-}
    if [[ -z "$cfg" ]]; then
        echo "[ERROR] Unsupported language code '${code}'." >&2
        exit 1
    fi
    echo "$cfg"
}

# Model aliases; override via env vars if needed (e.g., QWEN3_EMBEDDING_4B_ENCODER).
declare -A MODEL_ENCODERS=(
    [me5-large-instruct]="${ME5_ENCODER:-intfloat/multilingual-e5-large-instruct}"
    [gte-multilingual-base]="${GTE_MULTILINGUAL_BASE_ENCODER:-Alibaba-NLP/gte-multilingual-base}"
    [jina-embedding-v3]="${JINA_EMBEDDING_V3_ENCODER:-jinaai/jina-embeddings-v3}"
    [qwen3-embedding-0.6B]="${QWEN3_EMBEDDING_06B_ENCODER:-Qwen/Qwen3-Embedding-0.6B}"
    [qwen3-embedding-4B]="${QWEN3_EMBEDDING_4B_ENCODER:-Qwen/Qwen3-Embedding-4B}"
    [qwen3-embedding-8B]="${QWEN3_EMBEDDING_8B_ENCODER:-Qwen/Qwen3-Embedding-8B}"
)

# Plan definition ------------------------------------------------------------
COMPOSITION_PAIRS=(
    "en:ar"
    "en:zh"
    "de:nl"
    "en:de"
    "ar:zh"
    "zh:ru"
)
HUB_MONO_JOBS=(
    "zh:id:zh"  # doc lang : query lang A : query lang B
    "de:de:en"
)
SCRIPT_MONO_JOBS=(
    "ru:en:ru"
)
HIGH_SIGNAL_MONO_JOBS=(
    "ar:en:ar"  # EN–AR, AR docs
    "zh:en:zh"  # EN–ZH, ZH docs
    "de:de:nl"  # DE–NL, DE docs
    "zh:id:zh"  # ID–ZH, ZH docs
    "en:en:zh"  # EN–ZH, EN docs
    "de:de:en"  # DE–EN, DE docs
    "ar:ar:zh"  # AR–ZH, AR docs
    "zh:ar:zh"
    "zh:zh:ru"  # ZH–RU, ZH docs
    "ru:zh:ru"
    "en:en:ar"  # EN–AR, EN docs
    "en:en:de"
    "de:en:de"
)
SIZE_BILINGUAL_PAIRS=(
    "en:zh"  # EN–ZH, EN+ZH docs
    "de:nl"  # DE–NL, DE+NL docs
    "ar:zh"
    "zh:ru"
    "en:de"
)

CORE_MODELS=(
    "me5-large-instruct"
    "gte-multilingual-base"
    "jina-embedding-v3"
    "qwen3-embedding-0.6B"
)
SIZE_MODELS=(
    "qwen3-embedding-0.6B"
    "qwen3-embedding-4B"
    "qwen3-embedding-8B"
)

RUN_PHASE1=${RUN_PHASE1:-1}
RUN_PHASE2=${RUN_PHASE2:-1}
DEDUP_JOBS=${DEDUP_JOBS:-1}

# Helpers -------------------------------------------------------------------
require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Missing required file: $path" >&2
        exit 1
    fi
}

encoder_tag() {
    local enc=$1
    echo "${enc##*/}"
}

default_index_root_for_encoder() {
    local enc_tag=$1
    local base=${INDEX_ROOT_BASE:-$REPO_ROOT/indexes}
    local suffix=""
    if [[ -n "${INDEX_SUBSET_TAG:-}" ]]; then
        suffix="-${INDEX_SUBSET_TAG}"
    fi
    if [[ -n "${INDEX_ROOT_OVERRIDE:-}" ]]; then
        echo "$INDEX_ROOT_OVERRIDE"
    else
        echo "${base}/idx-${DATASET}-${enc_tag}${suffix}"
    fi
}

default_query_cache_root_for_encoder() {
    local enc_tag=$1
    local base=${QUERY_CACHE_ROOT_BASE:-$DATA_ROOT}
    if [[ -n "${QUERY_CACHE_ROOT_OVERRIDE:-}" ]]; then
        echo "$QUERY_CACHE_ROOT_OVERRIDE"
    else
        echo "${base}/ablation2/enc-query-${DATASET}-${enc_tag}"
    fi
}

declare -a JOB_QUEUE=()
declare -A SEEN_JOBS=()
declare -A NEEDED_QUERY_LANGS=()
declare -A BILINGUAL_LANG_SEEN=()
declare -A MONO_DOC_LANG_SEEN=()
declare -A ENCODER_QUERY_LANGS=()
ALLOW_OVERWRITE=${ALLOW_OVERWRITE:-1}

add_bilingual() {
    local model=$1
    local block=$2
    local lang_a=${3,,}
    local lang_b=${4,,}

    if [[ -z "${LANG_NAME_MAP[$lang_a]:-}" || -z "${LANG_NAME_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Unsupported language code in bilingual job '${lang_a}:${lang_b}' (block=${block}, model=${model})." >&2
        exit 1
    fi
    local key="${model}|bilingual|${lang_a}|${lang_b}"
    if (( DEDUP_JOBS )) && [[ -n "${SEEN_JOBS[$key]:-}" ]]; then
        echo "[INFO] Skipping duplicate job for ${key} (additional block=${block})" >&2
        return
    fi
    SEEN_JOBS[$key]=1
    NEEDED_QUERY_LANGS[$lang_a]=1
    NEEDED_QUERY_LANGS[$lang_b]=1
    BILINGUAL_LANG_SEEN[$lang_a]=1
    BILINGUAL_LANG_SEEN[$lang_b]=1
    JOB_QUEUE+=("${model},${block},bilingual,${lang_a},${lang_b}")
    local enc=${MODEL_ENCODERS[$model]:-}
    if [[ -n "$enc" ]]; then
        local enc_tag
        enc_tag=$(encoder_tag "$enc")
        local existing=${ENCODER_QUERY_LANGS[$enc_tag]:-}
        ENCODER_QUERY_LANGS[$enc_tag]="${existing} ${lang_a} ${lang_b}"
    fi
}

add_monolingual() {
    local model=$1
    local block=$2
    local doc_lang=${3,,}
    local lang_a=${4,,}
    local lang_b=${5,,}

    if [[ -z "${LANG_NAME_MAP[$doc_lang]:-}" ]]; then
        echo "[ERROR] Unsupported doc language '${doc_lang}' in monolingual job (block=${block}, model=${model})." >&2
        exit 1
    fi
    if [[ -z "${LANG_NAME_MAP[$lang_a]:-}" || -z "${LANG_NAME_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Unsupported query language in monolingual job ${doc_lang}:${lang_a}:${lang_b} (block=${block}, model=${model})." >&2
        exit 1
    fi

    local key="${model}|mono|${doc_lang}|${lang_a}|${lang_b}"
    if (( DEDUP_JOBS )) && [[ -n "${SEEN_JOBS[$key]:-}" ]]; then
        echo "[INFO] Skipping duplicate job for ${key} (additional block=${block})" >&2
        return
    fi
    SEEN_JOBS[$key]=1
    NEEDED_QUERY_LANGS[$lang_a]=1
    NEEDED_QUERY_LANGS[$lang_b]=1
    MONO_DOC_LANG_SEEN[$doc_lang]=1
    JOB_QUEUE+=("${model},${block},monolingual,${doc_lang},${lang_a},${lang_b}")
    local enc=${MODEL_ENCODERS[$model]:-}
    if [[ -n "$enc" ]]; then
        local enc_tag
        enc_tag=$(encoder_tag "$enc")
        local existing=${ENCODER_QUERY_LANGS[$enc_tag]:-}
        ENCODER_QUERY_LANGS[$enc_tag]="${existing} ${lang_a} ${lang_b}"
    fi
}

terminate_jobs() {
    for pid in "${!PID_TO_GPU[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
}
trap terminate_jobs SIGINT SIGTERM

gpu_has_capacity() {
    local gpu=$1
    local capacity=${GPU_CAPACITY[$gpu]:-$DEFAULT_GPU_SLOTS}
    local ours=${GPU_SLOT_USAGE[$gpu]:-0}
    if (( ours < capacity )); then
        return 0
    fi
    return 1
}

reap_finished_jobs() {
    for pid in "${!PID_TO_GPU[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            continue
        fi
        local status=0
        if ! wait "$pid"; then
            status=$?
        fi
        local desc=${PID_TO_DESC[$pid]}
        local log=${PID_TO_LOG[$pid]}
        local gpu=${PID_TO_GPU[$pid]}
        unset PID_TO_DESC["$pid"] PID_TO_LOG["$pid"] PID_TO_GPU["$pid"]
        if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
            GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] - 1))
        fi
        RUNNING_JOBS=$((RUNNING_JOBS - 1))
        if [[ $status -ne 0 ]]; then
            echo "[ERROR] Job '${desc}' failed with status ${status}. See ${log}" >&2
            terminate_jobs
            exit $status
        fi
        echo "[$(date '+%F %T')] Completed ${desc} (logs: ${log})" >&2
    done
}

wait_for_gpu_slot() {
    while true; do
        reap_finished_jobs
        for gpu in "${GPUS[@]}"; do
            if ! gpu_has_capacity "$gpu"; then
                continue
            fi
            echo "$gpu"
            return 0
        done
        echo "[$(date '+%F %T')] All GPU slots busy, retrying in ${SLEEP_INTERVAL}s..." >&2
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
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    echo "[$(date '+%F %T')] Launched ${desc} on GPU ${gpu} (log: ${log_file})" >&2
}

resolve_outdir() {
    local base=$1
    if [[ ! -d "$base" ]]; then
        echo "$base"
        return
    fi
    if ! find "$base" -mindepth 1 -print -quit | grep -q .; then
        echo "$base"
        return
    fi
    if (( ALLOW_OVERWRITE )); then
        echo "[WARN] Output directory exists and is non-empty, reusing due to ALLOW_OVERWRITE=1: $base" >&2
        echo "$base"
        return
    fi
    local n=1
    local candidate
    while true; do
        candidate="${base}-r${n}"
        if [[ ! -d "$candidate" ]] || ! find "$candidate" -mindepth 1 -print -quit | grep -q .; then
            echo "[INFO] Output directory ${base} exists; switching to ${candidate}" >&2
            echo "$candidate"
            return
        fi
        n=$((n + 1))
    done
}

lang_index_exists() {
    local root=$1
    local lang=$2
    [[ -f "${root}/${lang}/index.faiss" && -f "${root}/${lang}/docid_map.tsv" && -f "${root}/${lang}/docids.txt" ]]
}

run_bilingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3
    local encoder=$4
    local enc_tag=$5
    local index_root=$6
    local query_cache_root=$7
    local block=$8

    local doc_langs="${LANG_NAME_MAP[$lang_a]},${LANG_NAME_MAP[$lang_b]}"
    local lang_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}

    local exp_tag="bilingual-${lang_pair}-${block}"
    local rundir_base="${RUN_ROOT}/${DATASET}-${SIZE}-${exp_tag}-5bands-${enc_tag}/vector_mix"
    local rundir
    rundir=$(resolve_outdir "$rundir_base")/ablation2
    local docids_path="${rundir}/docids-${lang_pair}.txt"
    local result_dir_base="${RESULT_ROOT}/ablation2/${DATASET}-${SIZE}-${exp_tag}-5bands-${enc_tag}/vector_mix"
    local result_dir
    result_dir=$(resolve_outdir "$result_dir_base")
    mkdir -p "$rundir" "$result_dir" "$query_cache_root"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting ${lang_pair} (${doc_langs} docs) [${block}] with ${encoder}"

    "$PYTHON_BIN" "$BILINGUAL_SCRIPT" \
        --repo "$REPO" \
        --qrels_repo "BeIR/msmarco-qrels" \
        --langs "$doc_langs" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$encoder" \
        --device "cuda:${gpu}" \
        --docids_out "$docids_path" \
        --trust_remote \
        --batch 256 \
        --dtype fp16 \
        --max_docs "$SIZE" \
        --index_root "$index_root" \
        --query_tsv "${lang_a}=${q_primary}" \
        --query_tsv "${lang_b}=${q_secondary}" \
        --outdir "$rundir" \
        --cm_alphas "$CM_ALPHAS" \
        --cache_queries \
        --query_cache_dir "$query_cache_root"

    for band in "${BANDS[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --filter_qids "$COMMON_QIDS"
    done
}

run_monolingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3
    local doc_code=$4
    local encoder=$5
    local enc_tag=$6
    local index_root=$7
    local query_cache_root=$8
    local block=$9

    local doc_lang=${LANG_NAME_MAP[$doc_code]}
    local query_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}
    local exp_tag="mono-${doc_code}-${query_pair}-${block}"
    local rundir_base="${RUN_ROOT}/ablation2/${DATASET}-${SIZE}-${exp_tag}-5bands-${enc_tag}/vector_mix"
    local rundir
    rundir=$(resolve_outdir "$rundir_base")
    local docids_path="${rundir}/docids-${query_pair}.txt"
    local result_dir_base="${RESULT_ROOT}/ablation2/${DATASET}-${SIZE}-${exp_tag}-5bands-${enc_tag}/vector_mix"
    local result_dir
    result_dir=$(resolve_outdir "$result_dir_base")
    mkdir -p "$rundir" "$result_dir" "$query_cache_root"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting monolingual mix for ${doc_lang} docs (${query_pair} queries) [${block}] with ${encoder}"

    "$PYTHON_BIN" "$MONO_SCRIPT" \
        --repo "$REPO" \
        --config "collection-${doc_lang}" \
        --qrels_split "$QRELS_SPLIT" \
        --encoder "$encoder" \
        --device "cuda:${gpu}" \
        --run_out "$rundir" \
        --docids_out "$docids_path" \
        --trust_remote \
        --batch 256 \
        --qblock 256 \
        --dtype fp16 \
        --max_docs "$SIZE" \
        --index_root "$index_root" \
        --query_tsv "${lang_a}=${q_primary}" \
        --query_tsv "${lang_b}=${q_secondary}" \
        --cm_alphas "$CM_ALPHAS" \
        --cache_queries \
        --query_cache_dir "$query_cache_root"

    for band in "${BANDS[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --filter_qids "$COMMON_QIDS"
    done
}

launch_jobs() {
    for job in "${JOB_QUEUE[@]}"; do
        IFS=',' read -r model block job_type arg1 arg2 arg3 <<< "$job"
        local encoder=${MODEL_ENCODERS[$model]:-}
        if [[ -z "$encoder" ]]; then
            echo "[ERROR] Encoder not set for model '${model}'." >&2
            exit 1
        fi
        local enc_tag
        enc_tag=$(encoder_tag "$encoder")
        local index_root
        index_root=$(default_index_root_for_encoder "$enc_tag")
        local query_cache_root
        query_cache_root=$(default_query_cache_root_for_encoder "$enc_tag")

        local gpu desc log_file
        case "$job_type" in
            bilingual)
                local lang_a=$arg1
                local lang_b=$arg2
                local pair_slug="${lang_a}-${lang_b}"
                desc="${model} ${block} bilingual ${pair_slug}"
                gpu=$(wait_for_gpu_slot)
                log_file="${LOG_DIR}/${model}-${block}-bilingual-${pair_slug}-$(date '+%Y%m%d_%H%M%S')-$$.log"
                start_job "$gpu" "$desc" "$log_file" run_bilingual_job "$gpu" "$lang_a" "$lang_b" "$encoder" "$enc_tag" "$index_root" "$query_cache_root" "$block"
                ;;
            monolingual)
                local doc_lang=$arg1
                local lang_a=$arg2
                local lang_b=$arg3
                local pair_slug="${lang_a}-${lang_b}"
                desc="${model} ${block} mono ${doc_lang} (${pair_slug})"
                gpu=$(wait_for_gpu_slot)
                log_file="${LOG_DIR}/${model}-${block}-mono-${doc_lang}-${pair_slug}-$(date '+%Y%m%d_%H%M%S')-$$.log"
                start_job "$gpu" "$desc" "$log_file" run_monolingual_job "$gpu" "$lang_a" "$lang_b" "$doc_lang" "$encoder" "$enc_tag" "$index_root" "$query_cache_root" "$block"
                ;;
            *)
                echo "[ERROR] Unknown job type '${job_type}' in queue entry '${job}'." >&2
                exit 1
                ;;
        esac

        if (( BETWEEN_LAUNCH_SLEEP > 0 )); then
            sleep "$BETWEEN_LAUNCH_SLEEP"
        fi
    done
}

# Build the queue -----------------------------------------------------------
if (( RUN_PHASE1 )); then
    for model in "${CORE_MODELS[@]}"; do
        for pair in "${COMPOSITION_PAIRS[@]}"; do
            IFS=':' read -r lang_a lang_b <<< "$pair"
            add_bilingual "$model" "composition" "$lang_a" "$lang_b"
            add_monolingual "$model" "composition" "$lang_a" "$lang_a" "$lang_b"
            add_monolingual "$model" "composition" "$lang_b" "$lang_a" "$lang_b"
        done

        for spec in "${HUB_MONO_JOBS[@]}"; do
            IFS=':' read -r doc lang_a lang_b <<< "$spec"
            add_monolingual "$model" "hub" "$doc" "$lang_a" "$lang_b"
        done

        for spec in "${SCRIPT_MONO_JOBS[@]}"; do
            IFS=':' read -r doc lang_a lang_b <<< "$spec"
            add_monolingual "$model" "script" "$doc" "$lang_a" "$lang_b"
        done
    done
fi

if (( RUN_PHASE2 )); then
    for model in "${SIZE_MODELS[@]}"; do
        for pair in "${SIZE_BILINGUAL_PAIRS[@]}"; do
            IFS=':' read -r lang_a lang_b <<< "$pair"
            add_bilingual "$model" "size" "$lang_a" "$lang_b"
        done

        for spec in "${HIGH_SIGNAL_MONO_JOBS[@]}"; do
            IFS=':' read -r doc lang_a lang_b <<< "$spec"
            add_monolingual "$model" "size" "$doc" "$lang_a" "$lang_b"
        done
    done
fi

if (( ${#JOB_QUEUE[@]} == 0 )); then
    echo "[ERROR] No jobs queued. Check RUN_PHASE1/RUN_PHASE2 settings." >&2
    exit 1
fi

declare -A USED_ENCODERS=()
declare -A USED_ENCODER_MAP=()
for job in "${JOB_QUEUE[@]}"; do
    IFS=',' read -r model _block _type _a _b _c <<< "$job"
    encoder=${MODEL_ENCODERS[$model]:-}
    if [[ -z "$encoder" ]]; then
        echo "[ERROR] Encoder not set for model '${model}'." >&2
        exit 1
    fi
    enc_tag=$(encoder_tag "$encoder")
    USED_ENCODERS[$enc_tag]=1
    USED_ENCODER_MAP[$enc_tag]=$encoder
done

preencode_queries_all() {
    if (( ! PREENCODE_QUERIES )); then
        echo "[INFO] PREENCODE_QUERIES=0; skipping query caching." >&2
        return
    fi

    for enc_tag in "${!ENCODER_QUERY_LANGS[@]}"; do
        local encoder=${USED_ENCODER_MAP[$enc_tag]:-}
        if [[ -z "$encoder" ]]; then
            continue
        fi
        local cache_root
        cache_root=$(default_query_cache_root_for_encoder "$enc_tag")
        mkdir -p "$cache_root"

        # Deduplicate languages for this encoder
        local lang_str=${ENCODER_QUERY_LANGS[$enc_tag]}
        declare -A _seen=()
        local -a langs=()
        for lang in $lang_str; do
            [[ -z "$lang" ]] && continue
            if [[ -z "${_seen[$lang]:-}" ]]; then
                langs+=("$lang")
                _seen[$lang]=1
            fi
        done
        unset _seen
        if (( ${#langs[@]} == 0 )); then
            continue
        fi

        # Skip if all requested caches already exist for this encoder.
        local -a missing_langs=()
        for lang in "${langs[@]}"; do
            local cache_file="${cache_root}/${lang}/queries.npz"
            if [[ ! -f "$cache_file" ]]; then
                missing_langs+=("$lang")
            fi
        done
        if (( ${#missing_langs[@]} == 0 )); then
            echo "[INFO] Query caches already present for encoder=${enc_tag}; skipping." >&2
            continue
        fi

        # Partner selection for English caches
        local partner_for_en=""
        for l in "${langs[@]}"; do
            if [[ "$l" != "en" ]]; then
                partner_for_en=$l
                break
            fi
        done

        if [[ "$encoder" == "jinaai/jina-embeddings-v3" ]]; then
            local missing_dep
            missing_dep=$(REQS="einops" "$PYTHON_BIN" - <<'PY'
import importlib.util, os
reqs = os.environ.get("REQS", "").split()
missing = [r for r in reqs if importlib.util.find_spec(r) is None]
print(",".join(missing))
PY
)
            if [[ -n "$missing_dep" ]]; then
                echo "[ERROR] Missing Python deps for ${encoder}: ${missing_dep}. Install them and rerun pre-encoding." >&2
                exit 1
            fi
        fi

        for lang in "${missing_langs[@]}"; do
            local cache_file="${cache_root}/${lang}/queries.npz"
            if [[ -f "$cache_file" ]]; then
                echo "[INFO] Query cache exists for ${lang} at ${cache_file}; skipping." >&2
                continue
            fi

            local partner=""
            if [[ "$lang" == "en" ]]; then
                partner=$partner_for_en
            else
                partner="en"
            fi
            if [[ -z "$partner" || "$partner" == "$lang" || -z "${QUERY_FILE_MAP[$partner]:-}" ]]; then
                for cand in "${langs[@]}"; do
                    if [[ "$cand" != "$lang" ]]; then
                        partner=$cand
                        break
                    fi
                done
            fi
            if [[ -z "$partner" || "$partner" == "$lang" ]]; then
                echo "[WARN] Could not find partner for caching ${lang}; skipping pre-encode." >&2
                continue
            fi

            echo "[INFO] Caching queries for ${lang} (partner=${partner}) encoder=${enc_tag}"
            local -a cache_cmd=(
                "$PYTHON_BIN" "$CACHE_SCRIPT"
                --repo "$REPO"
                --encoder "$encoder"
                --device "$PREENC_QUERY_DEVICE"
                --enc_batch "$PREENC_QUERY_BATCH"
                --cache_root "$cache_root"
                --query_tsv "${partner}=${QUERY_FILE_MAP[$partner]}"
                --query_tsv "${lang}=${QUERY_FILE_MAP[$lang]}"
            )
            if (( PREENC_QUERY_TRUST_REMOTE )); then
                cache_cmd+=(--trust_remote)
            fi
            "${cache_cmd[@]}"
        done
    done
}

if [[ -n "${INDEX_ROOT_OVERRIDE:-}" && ${#USED_ENCODERS[@]} -gt 1 ]]; then
    echo "[WARN] INDEX_ROOT_OVERRIDE is set; sharing a single index root across multiple encoders (${!USED_ENCODERS[@]})." >&2
fi
if [[ -n "${QUERY_CACHE_ROOT_OVERRIDE:-}" && ${#USED_ENCODERS[@]} -gt 1 ]]; then
    echo "[WARN] QUERY_CACHE_ROOT_OVERRIDE is set; sharing a single query cache across multiple encoders (${!USED_ENCODERS[@]})." >&2
fi

# Sanity checks
require_file "$COMMON_QIDS"
require_file "$BILINGUAL_SCRIPT"
require_file "$MONO_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$CACHE_SCRIPT"
for lang in "${!NEEDED_QUERY_LANGS[@]}"; do
    require_file "${QUERY_FILE_MAP[$lang]}"
done

preencode_queries_all

launch_jobs
while (( RUNNING_JOBS > 0 )); do
    reap_finished_jobs
    if (( RUNNING_JOBS > 0 )); then
        sleep 60s
    fi
done

echo "[$(date '+%F %T')] All ablation experiments completed."
