#!/usr/bin/env bash
# ROADMAP step 2.1: sync code to the cluster and confirm the frozen hashes there.
# Run by the coordinator after G1 is recorded (step 1.3).
#
#   research/scripts/phase2_sync_check.sh                 # default: --dry-run
#   research/scripts/phase2_sync_check.sh --dry-run       # rsync -n, show what would change
#   research/scripts/phase2_sync_check.sh --apply [--other-jobs-ok REASON]
#   research/scripts/phase2_sync_check.sh --check-only    # no rsync, hash on the cluster
#   research/scripts/phase2_sync_check.sh --local-only    # steps 1 and 2 only, no SSH
#   Options: --strict-a6 (A6 pins must also match locally), --out PATH (JSON path)
#
# Steps:
#   1. Parse the pinned files and sha256 from the freeze sections of PREREG_A2, A4, A5
#      and A7 (frozen) and A6 (prepared, not frozen), via phase2_sync_helpers.py. Stop if
#      a freeze section does not parse or two frozen pins disagree.
#   2. Check every pinned file against the working tree. Stop on any frozen mismatch.
#      A6-only mismatches warn (A6 is not frozen) unless --strict-a6.
#   3. Over SSH (ControlMaster socket, BatchMode, never prompts): the remote project
#      root ~/rds/hpc-work/study3 must exist. rsync -n with --itemize-changes and
#      --checksum lists what would change; pinned files are flagged.
#   4. --apply only: squeue must show no s3_rotation job pending or running (no file
#      read by a running job may change), and no other active non-cas job unless
#      --other-jobs-ok REASON is given. If squeue fails, stop. The check runs again
#      immediately before the real rsync. Never --delete.
#   5. sha256 on the cluster for every pinned file, compared with the pins.
#      --apply and --check-only write research/reproduction/phase2_hash_check.json;
#      --dry-run writes to a temp file, since the cluster has not been synced.
#      Exit 1 on any frozen mismatch, 3 if only A6 pins differ, 0 if all match.
#
# Data, results, stimuli, media, caches and .git are excluded (list in the helper,
# `rsync-excludes`). Two A5 pins live under data/ and are hash-checked, not synced.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PYTHON:-/opt/anaconda3/envs/mr_eu_open_llm/bin/python}"
HELPER="${REPO}/research/scripts/phase2_sync_helpers.py"
HOST="${PHASE2_HOST:-eb2007@login.hpc.cam.ac.uk}"
REMOTE_REL="${PHASE2_REMOTE_REL:-rds/hpc-work/study3}"   # relative to the remote home
SSH_OPTS="-o ControlPath=/Users/eb2007/.ssh/cm-%r@%h:%p -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=10"
OUT="${REPO}/research/reproduction/phase2_hash_check.json"

MODE="dry-run"
OTHER_OK=""
STRICT_A6=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE="dry-run"; shift ;;
    --apply) MODE="apply"; shift ;;
    --check-only) MODE="check-only"; shift ;;
    --local-only) MODE="local-only"; shift ;;
    --strict-a6) STRICT_A6="--strict-a6"; shift ;;
    --other-jobs-ok) OTHER_OK="${2:-}"; [ -n "${OTHER_OK}" ] || { echo "--other-jobs-ok needs a REASON" >&2; exit 2; }; shift 2 ;;
    --out) OUT="${2:?--out needs a path}"; shift 2 ;;
    -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

die() { echo "STOP: $*" >&2; exit 1; }
py() { "${PYTHON}" "${HELPER}" --repo "${REPO}" "$@"; }
# shellcheck disable=SC2086
ssh_run() { perl -e 'alarm 45; exec @ARGV' ssh ${SSH_OPTS} "${HOST}" "$@"; }

[ -x "${PYTHON}" ] || die "python not found at ${PYTHON} (set PYTHON)"

echo "== 1. pinned files from the freeze sections"
py pins || die "freeze sections did not parse cleanly"

echo "== 2. local check against the working tree"
py local-check ${STRICT_A6} || die "local check failed; nothing synced"

if [ "${MODE}" = "local-only" ]; then
  echo "local-only: done, no SSH"
  exit 0
fi

command -v rsync >/dev/null || die "rsync not found"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/phase2_sync.XXXXXX")"
trap 'rm -rf "${WORK}"' EXIT

echo "== 3. cluster reachable and project root present"
ssh_run "test -d ${REMOTE_REL} && test -d ${REMOTE_REL}/scripts && echo ok" > "${WORK}/root.txt" \
  || die "SSH failed or ~/${REMOTE_REL} missing (re-authenticate CSD3 if the socket dropped)"
grep -qx ok "${WORK}/root.txt" || die "unexpected reply checking ~/${REMOTE_REL}"

EXCLUDES=()
while IFS= read -r pat; do EXCLUDES+=(--exclude "${pat}"); done < <(py rsync-excludes)
[ "${#EXCLUDES[@]}" -gt 0 ] || die "no rsync excludes loaded"
# rsync splits -e on spaces and honours single quotes. A longer alarm than for one-off
# commands, since the transfer runs inside this ssh.
RSH_DRY="perl -e 'alarm 300; exec @ARGV' ssh ${SSH_OPTS}"
RSH_APPLY="perl -e 'alarm 1800; exec @ARGV' ssh ${SSH_OPTS}"
RSYNC_FLAGS=(-rlpc --itemize-changes)   # no -t: identical content is left alone; never --delete

jobs_gate() {
  ssh_run "squeue -u eb2007 -h -o '%i|%j|%T'" > "${WORK}/squeue.txt" \
    || die "squeue failed over SSH; refusing to sync without a queue check"
  if [ -n "${OTHER_OK}" ]; then
    py jobs-gate --other-jobs-ok "${OTHER_OK}" < "${WORK}/squeue.txt"
  else
    py jobs-gate < "${WORK}/squeue.txt"
  fi
}

APPLIED=0
if [ "${MODE}" = "dry-run" ] || [ "${MODE}" = "apply" ]; then
  echo "== 3b. rsync dry run (nothing changes)"
  rsync -n "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" -e "${RSH_DRY}" \
    "${REPO}/" "${HOST}:${REMOTE_REL}/" > "${WORK}/rsync_dry.txt" \
    || die "rsync dry run failed"
  py annotate < "${WORK}/rsync_dry.txt"
fi

if [ "${MODE}" = "apply" ]; then
  echo "== 4. queue check before a real sync"
  jobs_gate || die "active jobs; no file read by a running job may change. Nothing synced."
  echo "== 4b. queue check again, then rsync (no --delete)"
  jobs_gate || die "queue changed; nothing synced"
  rsync "${RSYNC_FLAGS[@]}" "${EXCLUDES[@]}" -e "${RSH_APPLY}" \
    "${REPO}/" "${HOST}:${REMOTE_REL}/" | tee "${WORK}/rsync_apply.txt" \
    || die "rsync failed part-way; rerun --check-only to see the cluster state"
  APPLIED=1
fi

echo "== 5. sha256 on the cluster"
py remote-paths > "${WORK}/paths.txt"
# shellcheck disable=SC2016
ssh_run "cd ${REMOTE_REL} && while IFS= read -r f; do if [ -f \"\$f\" ]; then sha256sum -- \"\$f\"; else printf 'MISSING  %s\n' \"\$f\"; fi; done" \
  < "${WORK}/paths.txt" > "${WORK}/remote_hashes.txt" || die "remote hashing failed"
[ "$(wc -l < "${WORK}/remote_hashes.txt")" -eq "$(wc -l < "${WORK}/paths.txt")" ] \
  || die "remote returned $(wc -l < "${WORK}/remote_hashes.txt") lines for $(wc -l < "${WORK}/paths.txt") paths"

if [ "${MODE}" = "dry-run" ]; then
  OUT_USED="${WORK}/phase2_hash_check_dryrun.json"
  trap - EXIT   # keep the dry-run JSON for inspection
else
  OUT_USED="${OUT}"
fi
GIT_COMMIT="$(git -C "${REPO}" rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_DIRTY="$( [ -z "$(git -C "${REPO}" status --porcelain --untracked-files=no -- . 2>/dev/null)" ] && echo false || echo true)"
EXTRA="{\"git_commit\": \"${GIT_COMMIT}\", \"git_dirty_tracked\": ${GIT_DIRTY}, \"host\": \"${HOST}\"}"
APPLIED_FLAG=()
[ "${APPLIED}" -eq 1 ] && APPLIED_FLAG=(--rsync-applied)
set +e
# shellcheck disable=SC2088  # "~/..." is a label for the JSON, not a path to expand
py compare "${WORK}/remote_hashes.txt" --out "${OUT_USED}" --mode "${MODE}" \
  --remote-root "~/${REMOTE_REL}" --extra-json "${EXTRA}" ${APPLIED_FLAG[@]+"${APPLIED_FLAG[@]}"}
RC=$?
set -e
[ "${MODE}" = "dry-run" ] && echo "dry run: JSON at ${OUT_USED} (pre-sync state; not the logged check)"
exit "${RC}"
