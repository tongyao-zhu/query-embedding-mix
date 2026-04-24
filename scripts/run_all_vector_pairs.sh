#!/usr/bin/env bash
set -euo pipefail

# Run bilingual + monolingual vector experiments for a list of language pairs,
# distributing work across two GPUs and queueing jobs when devices are busy.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"
PYTHON_BIN=${PYTHON_BIN:-python}

DATASET=mmarco
REPO=unicamp-dl/mmarco
ENCODER=BAAI/bge-m3
ENC_TAG=${ENCODER##*/}
SIZE=8841823
QRELS_SPLIT=validation
CM_ALPHAS="0,0.1,0.3,0.5,0.7,0.9,1"
BANDS=(0 0.1 0.3 0.5 0.7 0.9 1)

DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/data}"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/runs}"
RUN_ROOT_EPHEMERAL=${RUN_ROOT_EPHEMERAL:-/tmp/mmarco_runs}
MIN_FREE_GB=${MIN_FREE_GB:-50}
USE_EPHEMERAL_RUNS=${USE_EPHEMERAL_RUNS:-0}
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/mmarco_full}"
QUERY_DIR="${QUERY_DIR:-$DATA_ROOT/mmarco_dev}"
COMMON_QIDS="${COMMON_QIDS:-$QUERY_DIR/queries_cm_5_bands_5-mini/qids-common.tsv}"
INDEX_ROOT="${INDEX_ROOT:-$REPO_ROOT/indexes/idx-${DATASET}-${ENC_TAG}-sub${SIZE}}"
QUERY_CACHE_ROOT="${QUERY_CACHE_ROOT:-$DATA_ROOT/enc-query-${DATASET}-${ENC_TAG}}"
QRELS_CACHE="${QRELS_CACHE:-$DATA_ROOT/qrels_cache}"

BILINGUAL_SCRIPT="${PY_ROOT}/onepass_bilingual_mix_hub_custom_lang.py"
MONO_SCRIPT="${PY_ROOT}/onepass_dense_mix_run_custom_lang.py"
EVAL_SCRIPT="${PY_ROOT}/evaluate.py"
CACHE_SCRIPT="${PY_ROOT}/cache_queries_for_mix.py"

LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/vector_pairs}"
mkdir -p "$LOG_DIR" "$RESULT_ROOT"

GPUS=(${GPUS:-0})
SLEEP_INTERVAL=${SLEEP_INTERVAL:-60}   # seconds between scheduler polls
BETWEEN_LAUNCH_SLEEP=${BETWEEN_LAUNCH_SLEEP:-5}   # fixed pause between launches
DEFAULT_GPU_SLOTS=1   # default concurrent jobs per GPU
BILINGUAL_PAIRS_FILE=${BILINGUAL_PAIRS_FILE:-${SCRIPT_DIR}/failed_pairs.txt}
MONO_JOBS_FILE=${MONO_JOBS_FILE:-${SCRIPT_DIR}/failed_monolingual_jobs.txt}
BILINGUAL_MAX_RUNNING=${BILINGUAL_MAX_RUNNING:-2}
MONO_MAX_RUNNING=${MONO_MAX_RUNNING:-4}
PAIR_MODE=${PAIR_MODE:-default}   # all | default | file
MONO_MODE=${MONO_MODE:-default} # pairs | default | file

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
LANG_CODES=(ar de en es fr hi id it ja nl pt ru vi zh)

declare -A QUERY_FILE_MAP
for code in "${!LANG_NAME_MAP[@]}"; do
    QUERY_FILE_MAP[$code]="${QUERY_DIR}/queries.${code}.tsv"
done

BILINGUAL_PAIRS_DEFAULT=(
    "en:fr"
    "en:it"
    "en:pt"
    "en:nl"
    "es:fr"
    "es:it"
    "fr:pt"
    "it:pt"
    "de:fr"
    "de:it"
    "nl:fr"
    "nl:it"
    "nl:es"
    "ja:hi"
    "ja:ru"
    "ar:zh"
    "hi:zh"
    "es:pt"
    "de:nl" 
    "en:de"
    "en:es" 
    "es:de"
    "en:id"
    "id:vi"
    "en:vi"
    "en:ru"
    "en:hi"
    "en:ar"
    "en:zh"
    "id:zh"
    "en:ja"
    "hi:ar"
    "fr:it"
    "zh:ja"
    "zh:ru"
)

MONO_JOBS_DEFAULT=(
    "en:en:fr"  # docLang:queryLangA:queryLangB
    "en:en:it"
    "en:en:pt"
    "en:en:nl"
    "fr:en:fr"
    "it:en:it"
    "pt:en:pt"
    "nl:en:nl"
    "es:es:fr"
    "fr:es:fr"
    "es:es:it"
    "it:es:it"
    "fr:fr:pt"
    "pt:fr:pt"
    "it:it:pt"
    "pt:it:pt"
    "de:de:fr"
    "fr:de:fr"
    "de:de:it"
    "it:de:it"
    "nl:nl:fr"
    "fr:nl:fr"
    "nl:nl:it"
    "it:nl:it"
    "nl:nl:es"
    "es:nl:es"
    "ja:ja:hi"
    "hi:ja:hi"
    "ja:ja:ru"
    "ru:ja:ru"
    "ar:ar:zh"
    "zh:ar:zh"
    "hi:hi:zh"
    "zh:hi:zh"
    "pt:es:pt"
    "nl:de:nl"
    "de:en:de"
    "es:en:es"
    "vi:id:vi"
    "zh:en:zh"
    "zh:id:zh"
    "ar:hi:ar"
    "ar:en:ar" 
    "de:es:de" 
    "de:de:nl" 
    "en:en:de" 
    "en:en:ar" 
    "en:en:es"
    "en:en:hi"
    "en:en:id"
    "en:en:zh"
    "en:en:ja"
    "en:en:ru"
    "en:en:vi"
    "es:es:de"
    "es:es:pt"
    "hi:en:hi"
    "hi:hi:ar"
    "id:en:id"
    "id:id:vi"
    "id:id:zh"
    "it:fr:it"
    "ja:en:ja"
    "ja:zh:ja"
    "ru:en:ru"
    "ru:zh:ru"
    "vi:en:vi"
    "zh:zh:ja"
    "zh:zh:ru"
    "fr:fr:it"
)

canonical_pair_key() {
    local a=$1
    local b=$2
    if [[ "$a" < "$b" ]]; then
        echo "${a}|${b}"
    else
        echo "${b}|${a}"
    fi
}

LANG_ORDER=(en es fr it de nl ja ar hi id zh pt ru vi)
declare -A LANG_ORDER_RANK=(
    [en]=0
    [es]=1
    [fr]=2
    [it]=3
    [de]=4
    [nl]=5
    [ja]=6
    [ar]=7
    [hi]=8
    [id]=9
    [zh]=10
    [pt]=11
    [ru]=12
    [vi]=13
)

order_pair_by_rank() {
    local lang_a=$1
    local lang_b=$2
    local rank_a=${LANG_ORDER_RANK[$lang_a]:-999}
    local rank_b=${LANG_ORDER_RANK[$lang_b]:-999}
    if (( rank_a <= rank_b )); then
        echo "${lang_a}:${lang_b}"
    else
        echo "${lang_b}:${lang_a}"
    fi
}

declare -a BILINGUAL_PAIRS=()
case "$PAIR_MODE" in
    file)
        if [[ -f "$BILINGUAL_PAIRS_FILE" ]]; then
            mapfile -t _pairs_from_file < <(grep -v '^\s*#' "$BILINGUAL_PAIRS_FILE" | sed '/^\s*$/d' | tr '[:upper:]' '[:lower:]')
            if (( ${#_pairs_from_file[@]} > 0 )); then
                BILINGUAL_PAIRS=("${_pairs_from_file[@]}")
                echo "[INFO] Loaded ${#BILINGUAL_PAIRS[@]} bilingual pairs from ${BILINGUAL_PAIRS_FILE}" >&2
            else
                echo "[WARN] ${BILINGUAL_PAIRS_FILE} exists but contained no pairs; falling back to generated pairs." >&2
            fi
        else
            echo "[WARN] ${BILINGUAL_PAIRS_FILE} not found; falling back to generated pairs." >&2
        fi
        ;;
    default)
        BILINGUAL_PAIRS=("${BILINGUAL_PAIRS_DEFAULT[@]}")
        ;;
    all|*)
        ;;
esac

if (( ${#BILINGUAL_PAIRS[@]} == 0 )); then
    for ((i=0; i<${#LANG_CODES[@]}; i++)); do
        for ((j=i+1; j<${#LANG_CODES[@]}; j++)); do
            lang_a=${LANG_CODES[$i]}
            lang_b=${LANG_CODES[$j]}
            BILINGUAL_PAIRS+=("$(order_pair_by_rank "$lang_a" "$lang_b")")
        done
    done
    echo "[INFO] Generated ${#BILINGUAL_PAIRS[@]} bilingual pairs (mode=${PAIR_MODE})." >&2
fi

declare -a MONO_JOBS=()
case "$MONO_MODE" in
    file)
        if [[ -f "$MONO_JOBS_FILE" ]]; then
            mapfile -t _mono_from_file < <(grep -v '^\s*#' "$MONO_JOBS_FILE" | sed '/^\s*$/d' | tr '[:upper:]' '[:lower:]')
            if (( ${#_mono_from_file[@]} > 0 )); then
                MONO_JOBS=("${_mono_from_file[@]}")
                echo "[INFO] Loaded ${#MONO_JOBS[@]} monolingual jobs from ${MONO_JOBS_FILE}" >&2
            else
                echo "[WARN] ${MONO_JOBS_FILE} exists but contained no jobs; falling back to pair-derived jobs." >&2
            fi
        else
            echo "[WARN] ${MONO_JOBS_FILE} not found; falling back to pair-derived jobs." >&2
        fi
        ;;
    default)
        MONO_JOBS=("${MONO_JOBS_DEFAULT[@]}")
        ;;
    pairs|*)
        ;;
esac

if (( ${#MONO_JOBS[@]} == 0 )); then
    for pair in "${BILINGUAL_PAIRS[@]}"; do
        IFS=':' read -r lang_a lang_b <<< "$pair"
        MONO_JOBS+=("${lang_a}:${lang_a}:${lang_b}")
        MONO_JOBS+=("${lang_b}:${lang_a}:${lang_b}")
    done
    echo "[INFO] Generated ${#MONO_JOBS[@]} monolingual jobs (mode=${MONO_MODE})." >&2
fi

require_file() {
    local path=$1
    if [[ ! -f "$path" ]]; then
        echo "[ERROR] Missing required file: $path" >&2
        echo "        Use download_mmarco_queries.py to pull the needed queries." >&2
        exit 1
    fi
}


disk_free_gb() {
    # Probe the filesystem backing RUN_ROOT; walk up until an existing path.
    local probe_path="$RUN_ROOT"
    while [[ ! -e "$probe_path" && "$probe_path" != "/" ]]; do
        probe_path=$(dirname "$probe_path")
    done
    local avail_kb
    avail_kb=$(df -Pk "$probe_path" 2>/dev/null | awk 'NR==2 {print $4}') || true
    if [[ -z "${avail_kb:-}" ]]; then
        echo ""
        return 0
    fi
    echo $((avail_kb / 1024 / 1024))
}

maybe_enable_ephemeral_runs() {
    if (( USE_EPHEMERAL_RUNS == 1 )); then
        return 0
    fi
    local free_gb
    free_gb=$(disk_free_gb)
    if [[ -z "${free_gb:-}" ]]; then
        return 0
    fi
    if (( free_gb < MIN_FREE_GB )); then
        USE_EPHEMERAL_RUNS=1
        echo "[WARN] Low disk (${free_gb} GiB < ${MIN_FREE_GB} GiB). Using ephemeral run storage at ${RUN_ROOT_EPHEMERAL} and cleaning after each job." >&2
    fi
}

band_result_exists() {
    local result_dir=$1
    local band=$2
    if compgen -G "${result_dir}/cm-alpha-${band}_dev_*-agg.json" > /dev/null; then
        return 0
    fi
    return 1
}

results_complete() {
    local result_dir=$1
    for band in "${BANDS[@]}"; do
        if ! band_result_exists "$result_dir" "$band"; then
            return 1
        fi
    done
    return 0
}

run_artifacts_complete() {
    local rundir=$1
    local docids_path=$2
    if [[ ! -f "$docids_path" ]]; then
        return 1
    fi
    for band in "${BANDS[@]}"; do
        if [[ ! -f "${rundir}/cm-alpha-${band}.trec" ]]; then
            return 1
        fi
    done
    return 0
}

bilingual_result_dir_for() {
    local lang_a=$1
    local lang_b=$2
    local exp_tag="bilingual-${lang_a}-${lang_b}"
    echo "${RESULT_ROOT}/${DATASET}-${SIZE}-${exp_tag}-5bands-${ENC_TAG}/vector_mix"
}

monolingual_result_dir_for() {
    local doc_code=$1
    local lang_a=$2
    local lang_b=$3
    local doc_lang=${LANG_NAME_MAP[$doc_code]}
    echo "${RESULT_ROOT}/${DATASET}-${SIZE}-${doc_lang}-${lang_a}-${lang_b}-5bands-${ENC_TAG}/vector_mix"
}

for lang in "${LANG_CODES[@]}"; do
    require_file "${QUERY_FILE_MAP[$lang]}"
done

require_file "$COMMON_QIDS"
require_file "$BILINGUAL_SCRIPT"
require_file "$MONO_SCRIPT"
require_file "$EVAL_SCRIPT"
require_file "$CACHE_SCRIPT"

declare -a normalize_pairs=()
declare -A pair_seen=()
for pair in "${BILINGUAL_PAIRS[@]}"; do
    IFS=':' read -r raw_a raw_b <<< "$pair"
    lang_a=${raw_a,,}
    lang_b=${raw_b,,}
    if [[ -z "${LANG_NAME_MAP[$lang_a]:-}" || -z "${LANG_NAME_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Unsupported language code in pair '${pair}'." >&2
        exit 1
    fi
    key=$(canonical_pair_key "$lang_a" "$lang_b")
    if [[ -n "${pair_seen[$key]:-}" ]]; then
        continue
    fi
    pair_seen[$key]=1
    normalize_pairs+=("${lang_a}:${lang_b}")
    require_file "${QUERY_FILE_MAP[$lang_a]}"
    require_file "${QUERY_FILE_MAP[$lang_b]}"
done
declare -a BILINGUAL_PAIRS=("${normalize_pairs[@]}")

declare -a normalized_mono_jobs=()
declare -A mono_seen=()
for job in "${MONO_JOBS[@]}"; do
    IFS=':' read -r raw_doc raw_a raw_b <<< "$job"
    doc_lang=${raw_doc,,}
    lang_a=${raw_a,,}
    lang_b=${raw_b,,}
    if [[ -z "$doc_lang" || -z "$lang_a" || -z "$lang_b" ]]; then
        echo "[ERROR] Bad monolingual job spec '${job}' (expected doc:langA:langB)" >&2
        exit 1
    fi
    if [[ -z "${LANG_NAME_MAP[$doc_lang]:-}" ]]; then
        echo "[ERROR] Unsupported document language code '${doc_lang}' in monolingual job '${job}'." >&2
        exit 1
    fi
    if [[ -z "${QUERY_FILE_MAP[$lang_a]:-}" || -z "${QUERY_FILE_MAP[$lang_b]:-}" ]]; then
        echo "[ERROR] Missing query TSV mapping for '${lang_a}' or '${lang_b}' in monolingual job '${job}'." >&2
        exit 1
    fi
    require_file "${QUERY_FILE_MAP[$lang_a]}"
    require_file "${QUERY_FILE_MAP[$lang_b]}"
    key="${doc_lang}:${lang_a}:${lang_b}"
    if [[ -n "${mono_seen[$key]:-}" ]]; then
        continue
    fi
    mono_seen[$key]=1
    normalized_mono_jobs+=("$key")
done
MONO_JOBS=("${normalized_mono_jobs[@]}")

declare -a JOB_QUEUE=()
for pair in "${BILINGUAL_PAIRS[@]}"; do
    IFS=':' read -r lang_a lang_b <<< "$pair"
    result_dir=$(bilingual_result_dir_for "$lang_a" "$lang_b")
    if results_complete "$result_dir"; then
        echo "[INFO] Skipping bilingual ${lang_a}-${lang_b}; results already complete." >&2
        continue
    fi
    JOB_QUEUE+=("bilingual,${lang_a},${lang_b}")
done

for job in "${MONO_JOBS[@]}"; do
    IFS=':' read -r doc_lang lang_a lang_b <<< "$job"
    result_dir=$(monolingual_result_dir_for "$doc_lang" "$lang_a" "$lang_b")
    if results_complete "$result_dir"; then
        echo "[INFO] Skipping monolingual ${doc_lang} (${lang_a}-${lang_b}); results already complete." >&2
        continue
    fi
    JOB_QUEUE+=("monolingual,${lang_a},${lang_b},${doc_lang}")
done

preencode_queries() {
    local device=${PREENCODE_DEVICE:-"cuda:0"}
    local batch=${PREENCODE_BATCH:-128}

    echo "[INFO] Pre-encoding queries for all languages into ${QUERY_CACHE_ROOT}"
    for lang in "${LANG_CODES[@]}"; do
        local cache_file="${QUERY_CACHE_ROOT}/${lang}/queries.npz"
        if [[ -f "$cache_file" ]]; then
            echo "[INFO] Cache exists for ${lang} (${cache_file}); skipping."
            continue
        fi
        echo "[INFO] Caching queries for ${lang}"
        "$PYTHON_BIN" "$CACHE_SCRIPT" \
            --repo "$REPO" \
            --encoder "$ENCODER" \
            --device "$device" \
            --enc_batch "$batch" \
            --cache_root "$QUERY_CACHE_ROOT" \
            --query_tsv "${lang}=${QUERY_FILE_MAP[$lang]}"
    done
}

declare -A GPU_CAPACITY=(
    [0]=${GPU0_SLOTS:-3}
    # [1]=${GPU1_SLOTS:-3}
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
declare -A PID_TO_PGID=()
declare -A PID_TO_TYPE=()
RUNNING_JOBS=0
RUNNING_BI=0
RUNNING_MONO=0

terminate_jobs() {
    for pid in "${!PID_TO_GPU[@]}"; do
        local pgid=${PID_TO_PGID[$pid]:-$pid}
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$pgid" == "$$" ]]; then
                kill -TERM "$pid" 2>/dev/null || true
                pkill -TERM -P "$pid" 2>/dev/null || true
            else
                kill -TERM -- "-${pgid}" 2>/dev/null || true
            fi
        fi
    done
    sleep 2
    for pid in "${!PID_TO_GPU[@]}"; do
        local pgid=${PID_TO_PGID[$pid]:-$pid}
        if kill -0 "$pid" 2>/dev/null; then
            if [[ "$pgid" == "$$" ]]; then
                kill -KILL "$pid" 2>/dev/null || true
                pkill -KILL -P "$pid" 2>/dev/null || true
            else
                kill -KILL -- "-${pgid}" 2>/dev/null || true
            fi
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
        local job_type=${PID_TO_TYPE[$pid]}
        unset PID_TO_DESC["$pid"] PID_TO_LOG["$pid"] PID_TO_GPU["$pid"] PID_TO_PGID["$pid"] PID_TO_TYPE["$pid"]
        if [[ ${GPU_SLOT_USAGE[$gpu]:-0} -gt 0 ]]; then
            GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] - 1))
        fi
        RUNNING_JOBS=$((RUNNING_JOBS - 1))
        if [[ "$job_type" == "bilingual" ]]; then
            RUNNING_BI=$((RUNNING_BI - 1))
        elif [[ "$job_type" == "monolingual" ]]; then
            RUNNING_MONO=$((RUNNING_MONO - 1))
        fi
        if [[ $status -ne 0 ]]; then
            echo "[ERROR] Job '${desc}' failed with status ${status}. See ${log}" >&2
            terminate_jobs
            exit $status
        fi
        echo "[$(date '+%F %T')] Completed ${desc} (logs: ${log})" >&2
    done
}

wait_for_gpu_slot() {
    local job_type=$1
    while true; do
        reap_finished_jobs
        if [[ "$job_type" == "bilingual" && $RUNNING_BI -ge $BILINGUAL_MAX_RUNNING ]]; then
            echo "[$(date '+%F %T')] Bilingual cap reached (${RUNNING_BI}/${BILINGUAL_MAX_RUNNING}); waiting..." >&2
            sleep "$SLEEP_INTERVAL"
            continue
        fi
        if [[ "$job_type" == "monolingual" && $RUNNING_MONO -ge $MONO_MAX_RUNNING ]]; then
            echo "[$(date '+%F %T')] Monolingual cap reached (${RUNNING_MONO}/${MONO_MAX_RUNNING}); waiting..." >&2
            sleep "$SLEEP_INTERVAL"
            continue
        fi
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
    local job_type=$4
    shift 4
    local pid
    local pgid
    (
        set -euo pipefail
        "$@"
    ) &> "$log_file" &
    pid=$!
    pgid=$pid
    PID_TO_GPU[$pid]=$gpu
    PID_TO_DESC[$pid]=$desc
    PID_TO_LOG[$pid]=$log_file
    PID_TO_PGID[$pid]=$pgid
    PID_TO_TYPE[$pid]=$job_type
    GPU_SLOT_USAGE[$gpu]=$((GPU_SLOT_USAGE[$gpu] + 1))
    RUNNING_JOBS=$((RUNNING_JOBS + 1))
    if [[ "$job_type" == "bilingual" ]]; then
        RUNNING_BI=$((RUNNING_BI + 1))
    elif [[ "$job_type" == "monolingual" ]]; then
        RUNNING_MONO=$((RUNNING_MONO + 1))
    fi
    echo "[$(date '+%F %T')] Launched ${desc} on GPU ${gpu} (log: ${log_file})" >&2
}

run_bilingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3

    local doc_lang_a=${LANG_NAME_MAP[$lang_a]}
    local doc_lang_b=${LANG_NAME_MAP[$lang_b]}
    local doc_langs="${doc_lang_a},${doc_lang_b}"
    local lang_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}
    local exp_tag="bilingual-${lang_pair}"
    local result_dir
    result_dir=$(bilingual_result_dir_for "$lang_a" "$lang_b")
    if results_complete "$result_dir"; then
        echo "[$(date '+%F %T')] [GPU ${gpu}] Skipping bilingual ${lang_pair}; results already complete." >&2
        return 0
    fi

    maybe_enable_ephemeral_runs
    local persistent_rundir="${RUN_ROOT}/${DATASET}-${SIZE}-${exp_tag}-5bands-${ENC_TAG}/vector_mix"
    local persistent_docids_path="${persistent_rundir}/docids-${lang_pair}.txt"
    local rundir="$persistent_rundir"
    local docids_path="$persistent_docids_path"
    local cleanup_root=""
    if (( USE_EPHEMERAL_RUNS == 1 )); then
        if ! run_artifacts_complete "$persistent_rundir" "$persistent_docids_path"; then
            mkdir -p "$RUN_ROOT_EPHEMERAL"
            cleanup_root=$(mktemp -d "${RUN_ROOT_EPHEMERAL}/${DATASET}-${SIZE}-${exp_tag}-XXXXXX")
            rundir="${cleanup_root}/vector_mix"
            docids_path="${rundir}/docids-${lang_pair}.txt"
        fi
    fi
    mkdir -p "$rundir" "$result_dir"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting bilingual mix for ${lang_pair} (${doc_langs})"

    local -a missing_bands=()
    for band in "${BANDS[@]}"; do
        if ! band_result_exists "$result_dir" "$band"; then
            missing_bands+=("$band")
        fi
    done

    if ! run_artifacts_complete "$rundir" "$docids_path"; then
        "$PYTHON_BIN" "$BILINGUAL_SCRIPT" \
            --repo "$REPO" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --langs "$doc_langs" \
            --qrels_split "$QRELS_SPLIT" \
            --encoder "$ENCODER" \
            --device "cuda:${gpu}" \
            --docids_out "$docids_path" \
            --trust_remote \
            --batch 1024 \
            --dtype fp16 \
            --max_docs "$SIZE" \
            --index_root "$INDEX_ROOT" \
            --query_tsv "${lang_a}=${q_primary}" \
            --query_tsv "${lang_b}=${q_secondary}" \
            --outdir "$rundir" \
            --cm_alphas "$CM_ALPHAS" \
            --cache_queries \
            --query_cache_dir "$QUERY_CACHE_ROOT"
    else
        echo "[$(date '+%F %T')] [GPU ${gpu}] Reusing existing bilingual run files for ${lang_pair}" >&2
    fi

    for band in "${missing_bands[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --perquery \
            --filter_qids "$COMMON_QIDS"
    done

    if ! run_artifacts_complete "$rundir" "$docids_path"; then
        echo "[ERROR] Bilingual ${lang_pair} missing run artifacts after completion; treating as failure." >&2
        return 1
    fi
    if ! results_complete "$result_dir"; then
        echo "[ERROR] Bilingual ${lang_pair} missing evaluation outputs after completion; treating as failure." >&2
        return 1
    fi

    if [[ -n "$cleanup_root" ]]; then
        if [[ "$cleanup_root" == "$RUN_ROOT_EPHEMERAL/"* ]]; then
            rm -rf "$cleanup_root"
        else
            echo "[WARN] Refusing to clean unexpected temp dir: ${cleanup_root}" >&2
        fi
    fi
}

run_monolingual_job() {
    local gpu=$1
    local lang_a=$2
    local lang_b=$3
    local doc_code=$4

    local doc_lang=${LANG_NAME_MAP[$doc_code]}
    local query_pair="${lang_a}-${lang_b}"
    local q_primary=${QUERY_FILE_MAP[$lang_a]}
    local q_secondary=${QUERY_FILE_MAP[$lang_b]}
    local result_dir
    result_dir=$(monolingual_result_dir_for "$doc_code" "$lang_a" "$lang_b")
    if results_complete "$result_dir"; then
        echo "[$(date '+%F %T')] [GPU ${gpu}] Skipping monolingual ${doc_lang} (${query_pair}); results already complete." >&2
        return 0
    fi

    maybe_enable_ephemeral_runs
    local persistent_rundir="${RUN_ROOT}/${DATASET}-${SIZE}-${doc_lang}-${query_pair}-5bands-${ENC_TAG}/vector_mix"
    local persistent_docids_path="${persistent_rundir}/docids-${query_pair}.txt"
    local rundir="$persistent_rundir"
    local docids_path="$persistent_docids_path"
    local cleanup_root=""
    if (( USE_EPHEMERAL_RUNS == 1 )); then
        if ! run_artifacts_complete "$persistent_rundir" "$persistent_docids_path"; then
            mkdir -p "$RUN_ROOT_EPHEMERAL"
            cleanup_root=$(mktemp -d "${RUN_ROOT_EPHEMERAL}/${DATASET}-${SIZE}-${doc_lang}-${query_pair}-XXXXXX")
            rundir="${cleanup_root}/vector_mix"
            docids_path="${rundir}/docids-${query_pair}.txt"
        fi
    fi
    mkdir -p "$rundir" "$result_dir"

    echo "[$(date '+%F %T')] [GPU ${gpu}] Starting monolingual mix for ${doc_lang} docs (${query_pair} queries)"

    local -a missing_bands=()
    for band in "${BANDS[@]}"; do
        if ! band_result_exists "$result_dir" "$band"; then
            missing_bands+=("$band")
        fi
    done

    if ! run_artifacts_complete "$rundir" "$docids_path"; then
        "$PYTHON_BIN" "$MONO_SCRIPT" \
            --repo "$REPO" \
            --config "collection-${doc_lang}" \
            --qrels_split "$QRELS_SPLIT" \
            --encoder "$ENCODER" \
            --device "cuda:${gpu}" \
            --run_out "$rundir" \
            --docids_out "$docids_path" \
            --trust_remote \
            --batch 1024 \
            --qblock 1024 \
            --dtype fp16 \
            --max_docs "$SIZE" \
            --index_root "$INDEX_ROOT" \
            --query_tsv "${lang_a}=${q_primary}" \
            --query_tsv "${lang_b}=${q_secondary}" \
            --cm_alphas "$CM_ALPHAS" \
            --cache_queries \
            --query_cache_dir "$QUERY_CACHE_ROOT"
    else
        echo "[$(date '+%F %T')] [GPU ${gpu}] Reusing existing monolingual run files for ${doc_lang} (${query_pair})" >&2
    fi

    for band in "${missing_bands[@]}"; do
        "$PYTHON_BIN" "$EVAL_SCRIPT" \
            --dataset "$DATASET" \
            --run "${rundir}/cm-alpha-${band}.trec" \
            --qrels_repo "BeIR/msmarco-qrels" \
            --qrels_split "$QRELS_SPLIT" \
            --outdir "$result_dir" \
            --trust_remote \
            --filter_docids "$docids_path" \
            --qrels_cache "$QRELS_CACHE" \
            --perquery \
            --filter_qids "$COMMON_QIDS"
    done

    if ! run_artifacts_complete "$rundir" "$docids_path"; then
        echo "[ERROR] Monolingual ${doc_lang} (${query_pair}) missing run artifacts after completion; treating as failure." >&2
        return 1
    fi
    if ! results_complete "$result_dir"; then
        echo "[ERROR] Monolingual ${doc_lang} (${query_pair}) missing evaluation outputs after completion; treating as failure." >&2
        return 1
    fi

    if [[ -n "$cleanup_root" ]]; then
        if [[ "$cleanup_root" == "$RUN_ROOT_EPHEMERAL/"* ]]; then
            rm -rf "$cleanup_root"
        else
            echo "[WARN] Refusing to clean unexpected temp dir: ${cleanup_root}" >&2
        fi
    fi
}

launch_jobs() {
    for job in "${JOB_QUEUE[@]}"; do
        IFS=',' read -r job_type lang_a lang_b doc_code <<< "$job"
        local pair_slug="${lang_a}-${lang_b}"
        local log_tag
        local desc
        case "$job_type" in
            bilingual)
                log_tag="bilingual-${pair_slug}"
                desc="bilingual ${pair_slug}"
                ;;
            monolingual)
                log_tag="monolingual-${doc_code}-${pair_slug}"
                desc="monolingual ${doc_code} (${pair_slug})"
                ;;
            *)
                echo "[ERROR] Unknown job type '${job_type}'." >&2
                exit 1
                ;;
        esac
        local gpu
        gpu=$(wait_for_gpu_slot "$job_type")
        local log_file="${LOG_DIR}/${log_tag}-$(date '+%Y%m%d_%H%M%S')-$$.log"
        if [[ "$job_type" == "bilingual" ]]; then
            start_job "$gpu" "$desc" "$log_file" "$job_type" run_bilingual_job "$gpu" "$lang_a" "$lang_b"
        else
            start_job "$gpu" "$desc" "$log_file" "$job_type" run_monolingual_job "$gpu" "$lang_a" "$lang_b" "$doc_code"
        fi
        if (( BETWEEN_LAUNCH_SLEEP > 0 )); then
            sleep "$BETWEEN_LAUNCH_SLEEP"
        fi
    done
}

# echo "[$(date '+%F %T')] Starting pre-encoding of queries..."
# preencode_queries
# echo "[$(date '+%F %T')] Finished pre-encoding queries."

launch_jobs
while (( RUNNING_JOBS > 0 )); do
    reap_finished_jobs
    if (( RUNNING_JOBS > 0 )); then
        sleep 60s
    fi
done

echo "[$(date '+%F %T')] All language-pair experiments completed."
