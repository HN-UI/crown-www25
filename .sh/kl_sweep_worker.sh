#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_ROOT}"

SCRIPT_TAG="${SCRIPT_TAG:-worker}"
GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEED="${SEED:-0}"
NUM_RUNS="${NUM_RUNS:-3}"
SEED_STEP="${SEED_STEP:-1}"
KL_TEMP="${KL_TEMP:-1.0}"
NEWS_ENCODER="${NEWS_ENCODER:-NAML}"
USER_ENCODER="${USER_ENCODER:-ATT}"
DATASET="${DATASET:-mind}"
MIND_SIZE="${MIND_SIZE:-small}"
RETRY_FAILED="${RETRY_FAILED:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "/home/dmais/anaconda3/envs/newsrec/bin/python" ]]; then
    PYTHON_BIN="/home/dmais/anaconda3/envs/newsrec/bin/python"
else
    PYTHON_BIN="python3"
fi

if [[ -n "${KL_WEIGHTS:-}" ]]; then
    read -r -a KL_WEIGHT_LIST <<< "${KL_WEIGHTS}"
else
    KL_WEIGHT_LIST=(0.1 0.2 0.4 0.6 0.8 1.0 1.5 2.0)
fi

if [[ "${DATASET}" == "mind" ]]; then
    DATASET_TAG="mind-${MIND_SIZE}"
else
    DATASET_TAG="${DATASET}"
fi

MODEL_NAME="${NEWS_ENCODER}-${USER_ENCODER}"
OUTPUT_PREFIX="${NEWS_ENCODER}_${USER_ENCODER}_KL"
STATE_ROOT="sweep_state/${NEWS_ENCODER,,}_${USER_ENCODER,,}_kl"
LOCK_ROOT="${STATE_ROOT}/locks"
DONE_ROOT="${STATE_ROOT}/done"
FAILED_ROOT="${STATE_ROOT}/failed"
WORKER_LOG="logs/${SCRIPT_TAG}_kl_worker.log"
CURRENT_LOCK_DIR=""
CURRENT_WEIGHT=""
ACTIVE_CHILD_PID=""

mkdir -p logs "${LOCK_ROOT}" "${DONE_ROOT}" "${FAILED_ROOT}"

log_msg() {
    local message
    message="[$(date '+%F %T')] [${SCRIPT_TAG}] $*"
    echo "${message}" | tee -a "${WORKER_LOG}"
}

preflight() {
    log_msg "python=${PYTHON_BIN}"
    if ! "${PYTHON_BIN}" -c "import torch, torchtext" >> "${WORKER_LOG}" 2>&1; then
        log_msg "preflight failed: torch/torchtext import failed for python=${PYTHON_BIN}"
        exit 2
    fi
}

safe_weight() {
    printf '%s' "${1//./p}"
}

exp_tag_for_weight() {
    printf 'kl_w_%s' "$(safe_weight "$1")"
}

result_dir_for_weight() {
    printf 'results/%s/%s/%s' "${DATASET_TAG}" "${MODEL_NAME}" "$(exp_tag_for_weight "$1")"
}

output_name_for_weight() {
    printf '%s_%s.tsv' "${OUTPUT_PREFIX}" "$1"
}

done_file_for_weight() {
    printf '%s/%s.done' "${DONE_ROOT}" "$(safe_weight "$1")"
}

failed_file_for_weight() {
    printf '%s/%s.failed' "${FAILED_ROOT}" "$(safe_weight "$1")"
}

release_current_lock() {
    if [[ -n "${CURRENT_LOCK_DIR}" && -d "${CURRENT_LOCK_DIR}" ]]; then
        rm -rf "${CURRENT_LOCK_DIR}"
    fi
    CURRENT_LOCK_DIR=""
    CURRENT_WEIGHT=""
}

terminate_active_child() {
    local child_pid=""

    child_pid="${ACTIVE_CHILD_PID}"
    if [[ -z "${child_pid}" ]]; then
        return
    fi

    if kill -0 "${child_pid}" 2>/dev/null; then
        log_msg "terminate active child pid=${child_pid}"
        pkill -TERM -P "${child_pid}" 2>/dev/null || true
        kill -TERM "${child_pid}" 2>/dev/null || true
        sleep 1
        pkill -KILL -P "${child_pid}" 2>/dev/null || true
        kill -KILL "${child_pid}" 2>/dev/null || true
    fi

    ACTIVE_CHILD_PID=""
}

cleanup() {
    terminate_active_child
    release_current_lock
}

handle_signal() {
    local signal_name="$1"
    log_msg "received ${signal_name}; shutting down"
    cleanup
    exit 128
}

trap cleanup EXIT
trap 'handle_signal HUP' HUP
trap 'handle_signal INT' INT
trap 'handle_signal TERM' TERM

count_completed_runs() {
    local result_dir="$1"
    local split_name="$2"
    local count=0
    local result_file

    if [[ ! -d "${result_dir}" ]]; then
        printf '0'
        return
    fi

    shopt -s nullglob
    for result_file in "${result_dir}"/#*-"${split_name}"; do
        if [[ -s "${result_file}" ]] && grep -q '[^[:space:]]' "${result_file}"; then
            count=$((count + 1))
        fi
    done
    shopt -u nullglob

    printf '%s' "${count}"
}

weight_done() {
    local weight="$1"
    local done_file
    local result_dir
    local dev_count
    local test_count

    done_file="$(done_file_for_weight "${weight}")"
    if [[ -f "${done_file}" ]]; then
        return 0
    fi

    result_dir="$(result_dir_for_weight "${weight}")"
    dev_count="$(count_completed_runs "${result_dir}" dev)"
    test_count="$(count_completed_runs "${result_dir}" test)"
    if (( dev_count >= NUM_RUNS && test_count >= NUM_RUNS )); then
        {
            printf 'weight=%s\n' "${weight}"
            printf 'script_tag=autodetected\n'
            printf 'gpu_id=unknown\n'
            printf 'completed_at=%s\n' "$(date '+%F %T')"
        } > "${done_file}"
        return 0
    fi

    return 1
}

lock_owner_alive() {
    local lock_dir="$1"
    local owner_pid=""

    if [[ ! -f "${lock_dir}/pid" ]]; then
        return 1
    fi

    read -r owner_pid < "${lock_dir}/pid" || true
    [[ -n "${owner_pid}" ]] || return 1
    kill -0 "${owner_pid}" 2>/dev/null
}

weight_failed() {
    local weight="$1"
    [[ -f "$(failed_file_for_weight "${weight}")" ]]
}

clear_failed_marker() {
    local failed_file
    failed_file="$(failed_file_for_weight "$1")"
    if [[ -f "${failed_file}" ]]; then
        rm -f "${failed_file}"
    fi
}

claim_next_weight() {
    local weight
    local safe_kl_w
    local lock_dir
    local owner_tag

    for weight in "${KL_WEIGHT_LIST[@]}"; do
        if weight_done "${weight}"; then
            log_msg "skip completed lambda=${weight}"
            continue
        fi

        if (( RETRY_FAILED == 0 )) && weight_failed "${weight}"; then
            log_msg "skip failed lambda=${weight} retry_failed=${RETRY_FAILED}"
            continue
        fi

        if (( RETRY_FAILED != 0 )); then
            clear_failed_marker "${weight}"
        fi

        safe_kl_w="$(safe_weight "${weight}")"
        lock_dir="${LOCK_ROOT}/${safe_kl_w}.lock"

        while true; do
            if mkdir "${lock_dir}" 2>/dev/null; then
                printf '%s\n' "$$" > "${lock_dir}/pid"
                printf '%s\n' "${SCRIPT_TAG}" > "${lock_dir}/script_tag"
                printf '%s\n' "${GPU_ID}" > "${lock_dir}/gpu_id"
                printf '%s\n' "${weight}" > "${lock_dir}/weight"
                printf '%s\n' "$(date '+%F %T')" > "${lock_dir}/claimed_at"
                CURRENT_LOCK_DIR="${lock_dir}"
                CURRENT_WEIGHT="${weight}"

                if weight_done "${weight}"; then
                    log_msg "lambda=${weight} finished while claiming; releasing lock"
                    release_current_lock
                    break
                fi

                log_msg "claimed lambda=${weight} on gpu=${GPU_ID}"
                return 0
            fi

            if ! lock_owner_alive "${lock_dir}"; then
                log_msg "remove stale lock lambda=${weight}"
                rm -rf "${lock_dir}"
                continue
            fi

            owner_tag="unknown"
            if [[ -f "${lock_dir}/script_tag" ]]; then
                read -r owner_tag < "${lock_dir}/script_tag" || true
            fi
            log_msg "skip locked lambda=${weight} owner=${owner_tag}"
            break
        done
    done

    return 1
}

mark_weight_done() {
    local weight="$1"
    local log_file="$2"
    local done_file

    done_file="$(done_file_for_weight "${weight}")"
    {
        printf 'weight=%s\n' "${weight}"
        printf 'script_tag=%s\n' "${SCRIPT_TAG}"
        printf 'gpu_id=%s\n' "${GPU_ID}"
        printf 'completed_at=%s\n' "$(date '+%F %T')"
        printf 'log_file=%s\n' "${log_file}"
    } > "${done_file}"
}

mark_weight_failed() {
    local weight="$1"
    local log_file="$2"
    local status="$3"
    local failed_file

    failed_file="$(failed_file_for_weight "${weight}")"
    {
        printf 'weight=%s\n' "${weight}"
        printf 'script_tag=%s\n' "${SCRIPT_TAG}"
        printf 'gpu_id=%s\n' "${GPU_ID}"
        printf 'failed_at=%s\n' "$(date '+%F %T')"
        printf 'status=%s\n' "${status}"
        printf 'log_file=%s\n' "${log_file}"
    } > "${failed_file}"
}

refresh_summary() {
    set +e
    "${PYTHON_BIN}" summarize_kl_sweep.py \
        --news-encoder "${NEWS_ENCODER}" \
        --user-encoder "${USER_ENCODER}" \
        --dataset "${DATASET}" \
        --mind-size "${MIND_SIZE}" \
        --num-runs "${NUM_RUNS}" \
        --weights "${KL_WEIGHT_LIST[@]}" \
        >> "${WORKER_LOG}" 2>&1
    set -e
}

run_claimed_weight() {
    local weight="$1"
    local exp_tag
    local output_name
    local log_file
    local result_dir
    local dev_count
    local test_count
    local status=0

    exp_tag="$(exp_tag_for_weight "${weight}")"
    output_name="$(output_name_for_weight "${weight}")"
    log_file="logs/${SCRIPT_TAG}_${output_name%.tsv}.log"
    result_dir="$(result_dir_for_weight "${weight}")"

    log_msg "launch lambda=${weight} gpu=${GPU_ID} exp_tag=${exp_tag} output=${output_name}"

    set +e
    "${PYTHON_BIN}" main.py \
        --news_encoder "${NEWS_ENCODER}" \
        --user_encoder "${USER_ENCODER}" \
        --device_id "${GPU_ID}" \
        --seed "${SEED}" \
        --num_runs "${NUM_RUNS}" \
        --seed_step "${SEED_STEP}" \
        --batch_size "${BATCH_SIZE}" \
        --dataset "${DATASET}" \
        --mind_size "${MIND_SIZE}" \
        --use_prev_nonclick_kl \
        --prev_nonclick_kl_weight "${weight}" \
        --prev_nonclick_kl_temperature "${KL_TEMP}" \
        --exp_tag "${exp_tag}" \
        --output-name "${output_name}" \
        > "${log_file}" 2>&1 &
    ACTIVE_CHILD_PID=$!
    wait "${ACTIVE_CHILD_PID}"
    status=$?
    ACTIVE_CHILD_PID=""
    set -e

    refresh_summary

    dev_count="$(count_completed_runs "${result_dir}" dev)"
    test_count="$(count_completed_runs "${result_dir}" test)"

    if (( status == 0 )) && (( dev_count >= NUM_RUNS )) && (( test_count >= NUM_RUNS )); then
        clear_failed_marker "${weight}"
        mark_weight_done "${weight}" "${log_file}"
        log_msg "done lambda=${weight} dev_runs=${dev_count} test_runs=${test_count}"
        return 0
    fi

    mark_weight_failed "${weight}" "${log_file}" "${status}"
    log_msg "failed lambda=${weight} status=${status} dev_runs=${dev_count} test_runs=${test_count} log=${log_file}"
    return 1
}

main() {
    local failures=0

    log_msg "worker start gpu=${GPU_ID} batch_size=${BATCH_SIZE} num_runs=${NUM_RUNS} kl_temp=${KL_TEMP}"
    preflight
    refresh_summary

    while claim_next_weight; do
        if ! run_claimed_weight "${CURRENT_WEIGHT}"; then
            failures=$((failures + 1))
        fi
        release_current_lock
    done

    refresh_summary

    if (( failures > 0 )); then
        log_msg "worker finished with failures=${failures}"
        exit 1
    fi

    log_msg "worker finished; no remaining lambda"
}

main "$@"
