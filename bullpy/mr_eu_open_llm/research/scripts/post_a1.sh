#!/usr/bin/env bash
# Post-A1 runbook (ROADMAP step 0.6). Run ONCE, by the coordinator, when all 30 A1
# rotation evals exist on the cluster.
#
#   research/scripts/post_a1.sh [--skip-slurm-check REASON]   # real run
#   research/scripts/post_a1.sh --dry-run DIR [--workdir W] [--n-perm N]
#
# Real run:
#   1. Over SSH: Slurm arrays 39982137 and 39982138 must be COMPLETED (all 5 tasks each,
#      none queued), and exactly 30 files must match
#      ~/rds/hpc-work/study3/results/baseline/eu_emotions/*/eval_v2_*rot*.json, namely the
#      30 names analyse_rotation.py expects. Otherwise it stops.
#   2. Records the remote sha256 of each file, pulls them unread into a staging dir,
#      verifies hashes, then moves them to results/baseline/eu_emotions/<model>/.
#   3. Runs research/scripts/analyse_rotation_run.py once: the frozen analyse_rotation.py
#      (hash checked) with errored trials given 0 p_target mass (DECISIONS), counts in
#      research/reproduction/rotation_errored_trials.json.
#   4. Runs same_clip_family_analysis.py --model-route for each model, arguments as in
#      research/STATE.md item 1.
#   5. Writes research/reproduction/a1_pull_manifest.json. Logs go to research/logs/.
# --skip-slurm-check REASON skips step 1's Slurm part (e.g. a task rerun under a new job
# id, judged by hand). The reason is recorded in the manifest. File checks still run.
# It edits no doc. Nothing here prints or inspects an eval file's contents.
#
# Dry run: no SSH. DIR holds synthetic files laid out as DIR/<model>/<eval name>. They are
# copied into a scratch tree (copies of research/scripts, symlinks to src, data and
# eu_au_measures.csv) so the frozen scripts, whose ROOT comes from their own location, see
# them at the expected relative paths. Outputs and the manifest go to the scratch dir.
# --n-perm is allowed in a dry run only, to keep tests short; the real run uses the default.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/opt/anaconda3/envs/mr_eu_open_llm/bin/python}"
HELPER="${REPO}/research/scripts/post_a1_helpers.py"
MODELS=(qwen3vl gemma4 molmo2)
ARRAYS="39982137,39982138"
REMOTE_ROOT='~/rds/hpc-work/study3'
REMOTE_REL='rds/hpc-work/study3'  # scp paths are relative to the remote home
REMOTE_GLOB="${REMOTE_ROOT}/results/baseline/eu_emotions/*/eval_v2_*rot*.json"
FROZEN_ANALYSE_SHA="98890c5bff65f13abeb0549b61fe414dc9fa30082b84bcb564671683ec12e24f"
HOST="eb2007@login.hpc.cam.ac.uk"
SSH_OPTS=(-o "ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p" -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=10)

DRY_DIR=""
WORKDIR=""
N_PERM=""
SKIP_SLURM=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_DIR="${2:?--dry-run needs DIR}"; shift 2 ;;
    --workdir) WORKDIR="${2:?--workdir needs a path}"; shift 2 ;;
    --n-perm) N_PERM="${2:?--n-perm needs a number}"; shift 2 ;;
    --skip-slurm-check) SKIP_SLURM="${2:-}"; [ -n "${SKIP_SLURM}" ] || { echo "--skip-slurm-check needs a REASON" >&2; exit 2; }; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "STOP: $*" >&2; exit 1; }
ssh_run() { perl -e 'alarm 45; exec @ARGV' ssh "${SSH_OPTS[@]}" "${HOST}" "$@"; }

[ -x "${PYTHON}" ] || die "python not found at ${PYTHON} (set PYTHON)"
py() { "${PYTHON}" "${HELPER}" "$@"; }

STARTED="$(date -u +%Y-%m-%dT%H:%M:%S+00:00)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
GIT_COMMIT="$(git -C "${REPO}" rev-parse HEAD)"
GIT_DIRTY=0
[ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no -- .)" ] || GIT_DIRTY=1

ANALYSE_SHA="$(py hash-file "${REPO}/research/scripts/analyse_rotation.py")"
[ "${ANALYSE_SHA}" = "${FROZEN_ANALYSE_SHA}" ] \
  || die "analyse_rotation.py sha256 ${ANALYSE_SHA} is not the frozen ${FROZEN_ANALYSE_SHA}"

if [ -n "${DRY_DIR}" ]; then
  # ---------------------------------------------------------------- dry run setup
  MODE="dry-run"
  DRY_DIR="$(cd "${DRY_DIR}" && pwd -P)"
  case "${DRY_DIR}/" in "$(cd "${REPO}" && pwd -P)/"*) die "dry-run DIR must be outside the repo" ;; esac
  if [ -z "${WORKDIR}" ]; then WORKDIR="$(mktemp -d "${TMPDIR:-/tmp}/post_a1_dry.XXXXXX")"; fi
  mkdir -p "${WORKDIR}"
  WORKDIR="$(cd "${WORKDIR}" && pwd -P)"
  case "${WORKDIR}/" in "$(cd "${REPO}" && pwd -P)/"*) die "--workdir must be outside the repo" ;; esac
  ROOT_FOR_ANALYSIS="${WORKDIR}/tree"
  RUN_DIR="${WORKDIR}/run"
  MANIFEST="${WORKDIR}/a1_pull_manifest.json"
  [ ! -e "${ROOT_FOR_ANALYSIS}" ] || die "${ROOT_FOR_ANALYSIS} exists; use a fresh --workdir"
  mkdir -p "${ROOT_FOR_ANALYSIS}/research/scripts" "${ROOT_FOR_ANALYSIS}/research/reproduction" \
           "${ROOT_FOR_ANALYSIS}/data" "${RUN_DIR}"
  # Copies, not symlinks: the scripts take ROOT from Path(__file__).resolve().
  cp "${REPO}"/research/scripts/*.py "${ROOT_FOR_ANALYSIS}/research/scripts/"
  ln -s "${REPO}/src" "${ROOT_FOR_ANALYSIS}/src"
  ln -s "${REPO}/data/eu_emotions_human_entropy.json" "${ROOT_FOR_ANALYSIS}/data/eu_emotions_human_entropy.json"
  ln -s "${REPO}/research/reproduction/eu_au_measures.csv" "${ROOT_FOR_ANALYSIS}/research/reproduction/eu_au_measures.csv"
  for f in analyse_rotation.py analyse_rotation_run.py same_clip_family_analysis.py same_clip_cue_analysis.py human_au_effects.py; do
    [ "$(py hash-file "${REPO}/research/scripts/$f")" = "$(py hash-file "${ROOT_FOR_ANALYSIS}/research/scripts/$f")" ] \
      || die "copy of $f differs from the repo"
  done
  SOURCE="dry-run:${DRY_DIR}"
else
  # ---------------------------------------------------------------- real run setup
  MODE="real"
  [ -z "${N_PERM}" ] || die "--n-perm is for dry runs only; the real run uses the default 4999"
  [ -z "${WORKDIR}" ] || die "--workdir is for dry runs only"
  ROOT_FOR_ANALYSIS="${REPO}"
  RUN_DIR="${REPO}/research/logs/post_a1_${STAMP}"
  MANIFEST="${REPO}/research/reproduction/a1_pull_manifest.json"
  SOURCE="${HOST}:${REMOTE_ROOT}"
  [ ! -e "${MANIFEST}" ] || die "${MANIFEST} exists: this runbook runs once. Move it aside to rerun."
  for out in research/reproduction/rotation_summary.json research/reproduction/rotation_errored_trials.json \
             research/reproduction/same_clip_family_model_{qwen3vl,gemma4,molmo2}.json; do
    [ ! -e "${REPO}/${out}" ] || die "${out} already exists: analysis looks done already"
  done
  mkdir -p "${RUN_DIR}"
fi

STAGING="${RUN_DIR}/staging"
CMDLOG="${RUN_DIR}/commands.jsonl"
SLURM_JSON="${RUN_DIR}/slurm.json"
FILES_JSON="${RUN_DIR}/files.json"
REMOTE_SHA="${RUN_DIR}/remote_sha256.txt"
REMOTE_COUNT=""
mkdir -p "${STAGING}"
: > "${CMDLOG}"
py expected > "${RUN_DIR}/expected.txt"

STATUS="started"
write_manifest() {
  local code=$?
  set +e
  local outputs=("${ROOT_FOR_ANALYSIS}/research/reproduction/rotation_summary.json"
                 "${ROOT_FOR_ANALYSIS}/research/reproduction/rotation_errored_trials.json")
  for m in "${MODELS[@]}"; do
    outputs+=("${ROOT_FOR_ANALYSIS}/research/reproduction/same_clip_family_model_${m}.json")
  done
  [ "${STATUS}" != "started" ] || STATUS="stopped before analysis"
  py manifest --out "${MANIFEST}" --mode "${MODE}" --started "${STARTED}" \
    --git-commit "${GIT_COMMIT}" --git-dirty "${GIT_DIRTY}" --analysis-root "${ROOT_FOR_ANALYSIS}" \
    --source "${SOURCE}" --slurm "${SLURM_JSON}" --remote-count "${REMOTE_COUNT}" \
    --files "${FILES_JSON}" --commands "${CMDLOG}" --status "${STATUS}" --exit-code "${code}" \
    --slurm-skip-reason "${SKIP_SLURM}" \
    --repo "${REPO}" \
    --scripts research/scripts/analyse_rotation.py research/scripts/analyse_rotation_run.py \
              research/scripts/same_clip_family_analysis.py \
              research/scripts/same_clip_cue_analysis.py research/scripts/human_au_effects.py \
              research/scripts/post_a1.sh research/scripts/post_a1_helpers.py \
    --outputs "${outputs[@]}"
  echo "run dir: ${RUN_DIR}"
  exit "${code}"
}
trap write_manifest EXIT

# -------------------------------------------------------------------- 1. checks
if [ -n "${SKIP_SLURM}" ]; then
  echo "== Slurm check SKIPPED: ${SKIP_SLURM}"
  "${PYTHON}" -c 'import json,sys; json.dump({"ok": None, "skipped": True, "reason": sys.argv[1], "message": "Slurm check skipped by flag"}, open(sys.argv[2], "w"), indent=1)' "${SKIP_SLURM}" "${SLURM_JSON}"
fi
if [ "${MODE}" = "real" ] && [ -z "${SKIP_SLURM}" ]; then
  echo "== checking Slurm arrays ${ARRAYS}"
  ssh_run "squeue -h -r -j ${ARRAYS} -o '%i|%T'" > "${RUN_DIR}/squeue.txt" 2>&1 || true
  if grep -q '|' "${RUN_DIR}/squeue.txt"; then
    die "arrays still in the queue: $(tr '\n' ' ' < "${RUN_DIR}/squeue.txt")"
  fi
  ssh_run "sacct -X -n -P -j ${ARRAYS} -o JobID,State" > "${RUN_DIR}/sacct.txt" \
    || die "sacct over SSH failed"
  py check-sacct "${RUN_DIR}/sacct.txt" --json-out "${SLURM_JSON}" \
    || die "Slurm arrays not all COMPLETED (see ${RUN_DIR}/sacct.txt); if a task was rerun under a new job id, judge it and use --skip-slurm-check REASON"
fi
if [ "${MODE}" = "real" ]; then
  echo "== counting remote files"
  REMOTE_COUNT="$(ssh_run "ls -1 ${REMOTE_GLOB} 2>/dev/null | wc -l" | tr -d '[:space:]')"
  [ "${REMOTE_COUNT}" = "30" ] || die "expected 30 remote files matching ${REMOTE_GLOB}, found ${REMOTE_COUNT}"

  echo "== remote sha256"
  REMOTE_LIST="$(sed "s|^|${REMOTE_ROOT}/|" "${RUN_DIR}/expected.txt" | tr '\n' ' ')"
  ssh_run "cd ${REMOTE_ROOT} && sha256sum $(tr '\n' ' ' < "${RUN_DIR}/expected.txt")" > "${REMOTE_SHA}" \
    || die "remote sha256sum failed: one of the 30 expected names is missing (${REMOTE_LIST})"
  [ "$(wc -l < "${REMOTE_SHA}" | tr -d ' ')" = "30" ] || die "remote sha256sum did not return 30 lines"

  echo "== pulling 30 files (unread)"
  while read -r rel; do
    mkdir -p "${STAGING}/$(dirname "${rel}")"
    perl -e 'alarm 900; exec @ARGV' scp -q -p "${SSH_OPTS[@]}" \
      "${HOST}:${REMOTE_REL}/${rel}" "${STAGING}/${rel}" || die "scp failed for ${rel}"
  done < "${RUN_DIR}/expected.txt"
else
  echo "== dry run: taking files from ${DRY_DIR}"
  : > "${REMOTE_SHA}"
  while read -r rel; do
    src="${DRY_DIR}/${rel#results/baseline/eu_emotions/}"
    [ -f "${src}" ] || die "dry-run file missing: ${src}"
    echo "$(py hash-file "${src}")  ${rel}" >> "${REMOTE_SHA}"
    mkdir -p "${STAGING}/$(dirname "${rel}")"
    cp -p "${src}" "${STAGING}/${rel}"
  done < "${RUN_DIR}/expected.txt"
  REMOTE_COUNT="$(find "${DRY_DIR}" -type f -name 'eval_v2_*rot*.json' | wc -l | tr -d ' ')"
  [ "${REMOTE_COUNT}" = "30" ] || die "expected 30 files in ${DRY_DIR}, found ${REMOTE_COUNT}"
  [ -n "${SKIP_SLURM}" ] || echo '{"ok": true, "message": "dry run: Slurm not checked", "states": {}}' > "${SLURM_JSON}"
fi

# -------------------------------------------------------------------- 2. verify and place
echo "== verifying hashes"
py verify "${REMOTE_SHA}" "${STAGING}" --json-out "${FILES_JSON}" || die "hash verification failed"
while read -r rel; do
  dest="${ROOT_FOR_ANALYSIS}/${rel}"
  mkdir -p "$(dirname "${dest}")"
  if [ -e "${dest}" ]; then
    [ "$(py hash-file "${dest}")" = "$(py hash-file "${STAGING}/${rel}")" ] \
      || die "${rel} already exists locally with a different hash; not overwriting"
    rm -f "${STAGING}/${rel}"
  else
    mv "${STAGING}/${rel}" "${dest}"
  fi
done < "${RUN_DIR}/expected.txt"
py verify "${REMOTE_SHA}" "${ROOT_FOR_ANALYSIS}" --json-out "${FILES_JSON}" \
  || die "hash verification failed after placing files"

# -------------------------------------------------------------------- 3-4. analysis
STATUS="analysis run"
FAILED=0
echo "== analyse_rotation_run.py (frozen analyse_rotation.py, errored-trial guard)"
py run --cwd "${ROOT_FOR_ANALYSIS}" "${CMDLOG}" analyse_rotation -- \
  "${PYTHON}" research/scripts/analyse_rotation_run.py \
  --output research/reproduction/rotation_summary.json \
  --errored-out research/reproduction/rotation_errored_trials.json || FAILED=1

PERM_ARGS=()
[ -z "${N_PERM}" ] || PERM_ARGS=(--n-perm "${N_PERM}")
for m in "${MODELS[@]}"; do
  echo "== model route: ${m}"
  py run --cwd "${ROOT_FOR_ANALYSIS}" "${CMDLOG}" "model_route_${m}" -- \
    "${PYTHON}" research/scripts/same_clip_family_analysis.py --model-route \
    --au-csv research/reproduction/eu_au_measures.csv --model "${m}" \
    --model-evals "results/baseline/eu_emotions/${m}/eval_v2_eu_emotions_${m}_video_only_seed42_rot*_real.json" \
    ${PERM_ARGS[@]+"${PERM_ARGS[@]}"} || FAILED=1
done

if [ "${FAILED}" = "0" ]; then
  STATUS="complete"
  echo "== done; outputs in ${ROOT_FOR_ANALYSIS}/research/reproduction"
else
  STATUS="analysis step failed"
  echo "STOP: an analysis step failed; see logs in ${RUN_DIR}" >&2
  exit 1
fi
