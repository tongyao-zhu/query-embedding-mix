#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
PY_ROOT="$REPO_ROOT/query_embedding_mix"

PYTHON_BIN=${PYTHON_BIN:-python}
GENERATOR=${GENERATOR:-$PY_ROOT/generate_cm_bands.py}
ENV_FILE=${ENV_FILE:-$REPO_ROOT/.env}
DATA_ROOT=${DATA_ROOT:-$REPO_ROOT/data/mmarco_dev}
GLOBAL_QID_LIST=${QID_LIST:-$REPO_ROOT/configs/qid_lists/en_zh_min.ge6.tsv}
MODEL=${MODEL:-openai/gpt-5-mini}
MODEL_TAG=${MODEL_TAG:-5-mini}
WORKERS=${WORKERS:-2}
MAX_TRIES=${MAX_TRIES:-2}
LOG_LEVEL=${LOG_LEVEL:-info}
HTTP_REFERER=${OPENROUTER_HTTP_REFERER:-}
X_TITLE=${OPENROUTER_X_TITLE:-}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}

QID_LIST_EN_ZH=${QID_LIST_EN_ZH:-$GLOBAL_QID_LIST}
QID_LIST_EN_VI=${QID_LIST_EN_VI:-$GLOBAL_QID_LIST}
QID_LIST_ZH_VI=${QID_LIST_ZH_VI:-$GLOBAL_QID_LIST}
QID_LIST_HI_ID=${QID_LIST_HI_ID:-$GLOBAL_QID_LIST}

BANDS=(0-20 20-40 40-60 60-80 80-100)

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

valid_cm_filename() {
  local name=$1
  [[ "$name" =~ ^queries-cm([0-9]+|[0-9]+-[0-9]+)\.tsv$ ]]
}

warn_qid_list_fallback() {
  local pair=$1
  local qid_list=$2
  if [[ "$pair" != "en-zh" && "$qid_list" == "$GLOBAL_QID_LIST" ]]; then
    log "Warning: ${pair} is using the shared QID_LIST fallback (${qid_list}). Override the pair-specific QID_LIST_<PAIR> env var if that is not the intended source."
  fi
}

validate_pair_output() {
  local pair=$1
  local out_dir=$2
  local file name
  local -a valid_files=()
  local expected_count=$(( ${#BANDS[@]} + 2 ))

  require_dir "$out_dir"
  require_file "$out_dir/qids-common.tsv"
  require_file "$out_dir/queries-cm0.tsv"
  require_file "$out_dir/queries-cm100.tsv"

  shopt -s nullglob
  for file in "$out_dir"/queries-cm*.tsv; do
    name=$(basename "$file")
    if valid_cm_filename "$name"; then
      valid_files+=("$file")
    fi
  done
  shopt -u nullglob

  if [[ ${#valid_files[@]} -lt $expected_count ]]; then
    echo "[ERROR] Expected at least ${expected_count} valid queries-cm*.tsv files in $out_dir, found ${#valid_files[@]}" >&2
    exit 1
  fi

  log "Validated ${pair} output at ${out_dir} (${#valid_files[@]} valid cm files, $(awk 'END {print NR}' "$out_dir/qids-common.tsv") qids-common rows)"
}

write_pure_bands_and_common_qids() {
  local src_tsv=$1
  local tgt_tsv=$2
  local qid_list=$3
  local out_dir=$4

  "$PYTHON_BIN" - "$src_tsv" "$tgt_tsv" "$qid_list" "$out_dir" <<'PY'
from pathlib import Path
import sys

src_path = Path(sys.argv[1])
tgt_path = Path(sys.argv[2])
qid_list_path = Path(sys.argv[3])
out_dir = Path(sys.argv[4])


def read_qid_filter(path: Path) -> set[str]:
    lines = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return set()
    header = [col.strip().lower() for col in lines[0].split("\t")]
    qid_col = 0
    has_header = False
    for idx, col in enumerate(header):
        if col in {"qid", "id"}:
            qid_col = idx
            has_header = True
            break

    keep: set[str] = set()
    for line in lines[1 if has_header else 0 :]:
        cols = line.split("\t")
        qid = cols[qid_col].strip() if qid_col < len(cols) else cols[0].strip()
        if qid:
            keep.add(qid)
    return keep


def read_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line or "\t" not in line:
                continue
            qid, text = line.split("\t", 1)
            qid = qid.strip()
            if qid:
                rows.append((qid, text.strip()))
    return rows


def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid, text in rows:
            f.write(f"{qid}\t{text}\n")


def is_valid_cm_file(path: Path) -> bool:
    name = path.name
    if not name.startswith("queries-cm") or not name.endswith(".tsv"):
        return False
    band = name[len("queries-cm") : -len(".tsv")]
    return band.isdigit() or ("-" in band and all(part.isdigit() for part in band.split("-", 1)))


keep_qids = read_qid_filter(qid_list_path)
src_rows = [row for row in read_rows(src_path) if row[0] in keep_qids]
tgt_rows = [row for row in read_rows(tgt_path) if row[0] in keep_qids]

write_rows(out_dir / "queries-cm0.tsv", src_rows)
write_rows(out_dir / "queries-cm100.tsv", tgt_rows)

qid_sets: list[set[str]] = []
for path in sorted(out_dir.glob("queries-cm*.tsv")):
    if not is_valid_cm_file(path):
        continue
    qids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            qid = line.split("\t", 1)[0].strip()
            if qid:
                qids.add(qid)
    qid_sets.append(qids)

common = set.intersection(*qid_sets) if qid_sets else set()
with (out_dir / "qids-common.tsv").open("w", encoding="utf-8") as f:
    for qid in sorted(common, key=lambda x: (len(x), x)):
        f.write(f"{qid}\n")

print(f"[i] wrote {out_dir / 'queries-cm0.tsv'} ({len(src_rows)} rows)")
print(f"[i] wrote {out_dir / 'queries-cm100.tsv'} ({len(tgt_rows)} rows)")
print(f"[i] wrote {out_dir / 'qids-common.tsv'} ({len(common)} qids)")
PY
}

run_pair() {
  local src_lang=$1
  local tgt_lang=$2
  local out_dir=$3
  local qid_list=$4
  local src_tsv="$DATA_ROOT/queries.${src_lang}.tsv"
  local tgt_tsv="$DATA_ROOT/queries.${tgt_lang}.tsv"
  local pair="${src_lang}-${tgt_lang}"

  require_file "$GENERATOR"
  require_file "$qid_list"
  require_file "$src_tsv"
  require_file "$tgt_tsv"
  warn_qid_list_fallback "$pair" "$qid_list"

  log "Generating ${src_lang}-${tgt_lang} word-mix into ${out_dir}"

  local -a cmd=(
    "$PYTHON_BIN" "$GENERATOR"
    --src "$src_tsv"
    --tgt "$tgt_tsv"
    --src_lang "$src_lang"
    --tgt_lang "$tgt_lang"
    --out_dir "$out_dir"
    --qid_list "$qid_list"
    --bands "${BANDS[@]}"
    --model "$MODEL"
    --workers "$WORKERS"
    --max_tries "$MAX_TRIES"
    --log "$LOG_LEVEL"
    --env_file "$ENV_FILE"
  )

  if [[ -n "${MAX_ROWS:-}" ]]; then
    cmd+=(--max_rows "$MAX_ROWS")
  fi
  if [[ -n "$HTTP_REFERER" ]]; then
    cmd+=(--http_referer "$HTTP_REFERER")
  fi
  if [[ -n "$X_TITLE" ]]; then
    cmd+=(--x_title "$X_TITLE")
  fi

  "${cmd[@]}"
  write_pure_bands_and_common_qids "$src_tsv" "$tgt_tsv" "$qid_list" "$out_dir"
  validate_pair_output "$pair" "$out_dir"
}

OUT_DIR_EN_ZH=${OUT_DIR_EN_ZH:-$DATA_ROOT/queries_cm_5_bands_5-mini}
OUT_DIR_EN_VI=${OUT_DIR_EN_VI:-$DATA_ROOT/queries_cm_5_bands_en_vi_${MODEL_TAG}}
OUT_DIR_ZH_VI=${OUT_DIR_ZH_VI:-$DATA_ROOT/queries_cm_5_bands_zh_vi_${MODEL_TAG}}
OUT_DIR_HI_ID=${OUT_DIR_HI_ID:-$DATA_ROOT/queries_cm_5_bands_hi_id_${MODEL_TAG}}

if [[ "$VALIDATE_ONLY" -eq 1 ]]; then
  validate_pair_output "en-zh" "$OUT_DIR_EN_ZH"
  validate_pair_output "en-vi" "$OUT_DIR_EN_VI"
  validate_pair_output "zh-vi" "$OUT_DIR_ZH_VI"
  validate_pair_output "hi-id" "$OUT_DIR_HI_ID"
  log "All requested word-mix outputs validated."
  exit 0
fi

# Comment out any run_pair line below if you do not want to regenerate that pair.
run_pair "en" "zh" "$OUT_DIR_EN_ZH" "$QID_LIST_EN_ZH"
run_pair "en" "vi" "$OUT_DIR_EN_VI" "$QID_LIST_EN_VI"
run_pair "zh" "vi" "$OUT_DIR_ZH_VI" "$QID_LIST_ZH_VI"
run_pair "hi" "id" "$OUT_DIR_HI_ID" "$QID_LIST_HI_ID"

log "All requested word-mix generations completed."
