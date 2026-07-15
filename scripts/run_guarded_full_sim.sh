#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_GUARD_CONFIG="$ROOT/configs/geant4/variance_reduction_external_no_isaac_32threads_cpu_guarded.json"
DEFAULT_SOURCE_CONFIG="$ROOT/results/ral_ablation/sources/mix9_multi_isotope_cardinality_seed_2026061201.json"
DEFAULT_OUTPUT_TAG="mix9_multi_isotope_cardinality_proposed_seed_2026061201_guarded"

usage() {
    cat <<'USAGE'
Usage:
  scripts/run_guarded_full_sim.sh [--session NAME] [--log-dir DIR] [--config PATH] [--no-tmux] [-- MAIN_ARGS...]

Runs a Geant4 full simulation in a persistent tmux session with host/GPU
heartbeat logging, CPU-only PF/planning by default, and a user systemd scope
when available.

If MAIN_ARGS are omitted, the June 2026 RA-L mix9 proposed run is used with a
guarded output tag. If MAIN_ARGS do not include --sim-config, the CPU-guarded
Geant4 config is inserted.

Environment overrides:
  RAL_MEMORY_MAX=96G       systemd MemoryMax for the run
  RAL_CPU_QUOTA=1600%      systemd CPUQuota for the run
  RAL_NATIVE_THREADS=16    BLAS/OpenMP thread cap
  RAL_ALLOW_CUDA=1         keep CUDA_VISIBLE_DEVICES instead of disabling CUDA
  RAL_SYSTEMD_SCOPE=0      run without systemd-run --user --scope
USAGE
}

has_arg() {
    local needle="$1"
    shift
    local arg
    for arg in "$@"; do
        if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
            return 0
        fi
    done
    return 1
}

worker_main() {
    local session=""
    local main_log=""
    local monitor_log=""
    local meta_file=""
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --session)
                session="$2"
                shift 2
                ;;
            --main-log)
                main_log="$2"
                shift 2
                ;;
            --monitor-log)
                monitor_log="$2"
                shift 2
                ;;
            --meta)
                meta_file="$2"
                shift 2
                ;;
            --)
                shift
                break
                ;;
            *)
                echo "Unknown worker option: $1" >&2
                exit 2
                ;;
        esac
    done
    if [[ -z "$session" || -z "$main_log" || -z "$monitor_log" || -z "$meta_file" || $# -eq 0 ]]; then
        echo "Missing worker inputs." >&2
        exit 2
    fi

    mkdir -p "$(dirname "$main_log")"
    {
        echo "session=$session"
        echo "main_log=$main_log"
        echo "monitor_log=$monitor_log"
        echo "pid=$$"
        echo "started_at=$(date --iso-8601=seconds)"
        printf 'command='
        printf '%q ' "$@"
        echo
    } >> "$meta_file"

    monitor_loop() {
        while true; do
            echo "==== $(date --iso-8601=seconds) heartbeat ===="
            uptime || true
            free -m || true
            df -h "$ROOT" || true
            ps -eo pid,ppid,stat,pcpu,pmem,rss,comm,args --sort=-rss | head -n 25 || true
            if command -v nvidia-smi >/dev/null 2>&1; then
                nvidia-smi --query-gpu=timestamp,name,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits || true
            fi
            sleep 30
        done
    }

    monitor_loop >> "$monitor_log" 2>&1 &
    local monitor_pid=$!
    cleanup() {
        kill "$monitor_pid" >/dev/null 2>&1 || true
        wait "$monitor_pid" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT INT TERM

    local cuda_visible=""
    if [[ "${RAL_ALLOW_CUDA:-0}" == "1" ]]; then
        cuda_visible="${CUDA_VISIBLE_DEVICES-}"
    fi

    local -a scope_prefix=()
    if [[ "${RAL_SYSTEMD_SCOPE:-1}" != "0" ]] \
        && command -v systemd-run >/dev/null 2>&1 \
        && systemctl --user show-environment >/dev/null 2>&1; then
        scope_prefix=(
            systemd-run --user --scope --same-dir --quiet
            -p "MemoryMax=${RAL_MEMORY_MAX:-96G}"
            -p "MemorySwapMax=${RAL_MEMORY_SWAP_MAX:-0}"
            -p "CPUQuota=${RAL_CPU_QUOTA:-1600%}"
            -p "TasksMax=${RAL_TASKS_MAX:-4096}"
            --
        )
    fi

    local status=0
    {
        echo "==== guarded full simulation start $(date --iso-8601=seconds) ===="
        echo "session=$session"
        echo "monitor_log=$monitor_log"
        echo "systemd_scope=$([[ ${#scope_prefix[@]} -gt 0 ]] && echo yes || echo no)"
        echo "cuda_visible_devices=${cuda_visible:-<disabled>}"
        "${scope_prefix[@]}" env \
            PYTHONUNBUFFERED=1 \
            PYTHONFAULTHANDLER=1 \
            MALLOC_ARENA_MAX=4 \
            MPLBACKEND=Agg \
            CUDA_VISIBLE_DEVICES="$cuda_visible" \
            OMP_NUM_THREADS="${RAL_NATIVE_THREADS:-16}" \
            OPENBLAS_NUM_THREADS="${RAL_NATIVE_THREADS:-16}" \
            MKL_NUM_THREADS="${RAL_NATIVE_THREADS:-16}" \
            NUMEXPR_NUM_THREADS="${RAL_NATIVE_THREADS:-16}" \
            stdbuf -oL -eL "$@"
    } >> "$main_log" 2>&1 || status=$?
    echo "finished_at=$(date --iso-8601=seconds)" >> "$meta_file"
    echo "exit_status=$status" >> "$meta_file"
    exit "$status"
}

if [[ "${1-}" == "--worker" ]]; then
    shift
    worker_main "$@"
fi

SESSION="ral_full_guarded_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/logs/full_sim"
GUARD_CONFIG="$DEFAULT_GUARD_CONFIG"
USE_TMUX=1
MAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --session)
            SESSION="$2"
            shift 2
            ;;
        --log-dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --config)
            GUARD_CONFIG="$2"
            shift 2
            ;;
        --no-tmux)
            USE_TMUX=0
            shift
            ;;
        --)
            shift
            MAIN_ARGS=("$@")
            break
            ;;
        *)
            MAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#MAIN_ARGS[@]} -eq 0 ]]; then
    MAIN_ARGS=(
        --full-simulation
        --environment-mode random
        --obstacle-seed 2026061201
        --source-config "$DEFAULT_SOURCE_CONFIG"
        --birth
        --measurement-time-s 30
        --output-tag "$DEFAULT_OUTPUT_TAG"
    )
fi

if ! has_arg "--sim-config" "${MAIN_ARGS[@]}"; then
    MAIN_ARGS=(--sim-config "$GUARD_CONFIG" "${MAIN_ARGS[@]}")
fi

mkdir -p "$LOG_DIR"
MAIN_LOG="$LOG_DIR/${SESSION}.log"
MONITOR_LOG="$LOG_DIR/${SESSION}.monitor.log"
META_FILE="$LOG_DIR/${SESSION}.meta"
RUN_CMD=(uv run python main.py "${MAIN_ARGS[@]}")

{
    echo "session=$SESSION"
    echo "log=$MAIN_LOG"
    echo "monitor_log=$MONITOR_LOG"
    echo "meta=$META_FILE"
    echo "workdir=$ROOT"
    echo "launcher_started_at=$(date --iso-8601=seconds)"
    printf 'command='
    printf '%q ' "${RUN_CMD[@]}"
    echo
} > "$META_FILE"

if [[ "$USE_TMUX" == "1" && -n "${TMUX-}" ]]; then
    echo "Already inside tmux; running guarded simulation in this pane."
    exec "$0" --worker --session "$SESSION" --main-log "$MAIN_LOG" --monitor-log "$MONITOR_LOG" --meta "$META_FILE" -- "${RUN_CMD[@]}"
fi

if [[ "$USE_TMUX" == "1" ]] && command -v tmux >/dev/null 2>&1; then
    printf -v WORKER_COMMAND '%q ' "$0" --worker --session "$SESSION" --main-log "$MAIN_LOG" --monitor-log "$MONITOR_LOG" --meta "$META_FILE" -- "${RUN_CMD[@]}"
    tmux new-session -d -s "$SESSION" "$WORKER_COMMAND"
    echo "Started guarded full simulation in tmux session: $SESSION"
    echo "Main log: $MAIN_LOG"
    echo "Monitor log: $MONITOR_LOG"
    echo "Attach: tmux attach -t $SESSION"
    exit 0
fi

echo "tmux unavailable or disabled; running guarded simulation in the current shell."
exec "$0" --worker --session "$SESSION" --main-log "$MAIN_LOG" --monitor-log "$MONITOR_LOG" --meta "$META_FILE" -- "${RUN_CMD[@]}"
