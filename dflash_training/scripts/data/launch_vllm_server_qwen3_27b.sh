#!/usr/bin/env bash
set -euo pipefail

# Requires vLLM to be installed separately (not in requirements.txt):
#   pip install vllm
# See https://docs.vllm.ai/en/latest/getting_started/installation.html
#
# Qwen3.6-27B-FP8 (~27GB weights) requires ~30GB VRAM per worker.
# With 32GB GPUs, use 1 worker per GPU with tight memory budget.
# If OOM, reduce max_model_len or num_workers.

model_path=/root/Qwen3.6-27B-FP8
# 27B 模型需要 TP=2 (每 worker 2 张 GPU)
num_workers=4
tensor_parallel_size=2
start_port=30000
host=0.0.0.0
dtype=auto
gpu_memory_utilization=0.95
max_model_len=8192
log_dir=logs/vllm_qwen3_27b
heartbeat_interval=300

get_host_ip() {
    local host_ip=""

    if command -v hostname > /dev/null 2>&1; then
        host_ip=$(hostname -I 2> /dev/null | awk '{print $1}')
    fi

    if [[ -z "${host_ip}" ]] && command -v ip > /dev/null 2>&1; then
        host_ip=$(
            ip -4 route get 1.1.1.1 2> /dev/null | awk '
                /src/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "src") {
                            print $(i + 1)
                            exit
                        }
                    }
                }
            '
        )
    fi

    if [[ -z "${host_ip}" ]]; then
        host_ip=127.0.0.1
    fi

    printf '%s\n' "${host_ip}"
}

mkdir -p "${log_dir}"

host_ip=$(get_host_ip)
pids=()
ports=()
heartbeat_pid=""

print_heartbeat() {
    local timestamp alive_count status pid port
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    alive_count=0

    echo "[${timestamp}] heartbeat"
    for ((idx = 0; idx < ${#pids[@]}; idx++)); do
        pid=${pids[$idx]}
        port=${ports[$idx]}
        status=dead
        if kill -0 "${pid}" > /dev/null 2>&1; then
            status=alive
            alive_count=$((alive_count + 1))
        fi
        echo "  worker_index=${idx} pid=${pid} port=${port} status=${status}"
    done
    echo "  alive_workers=${alive_count}/${#pids[@]}"
}

heartbeat_loop() {
    while true; do
        sleep "${heartbeat_interval}"
        print_heartbeat
    done
}

cleanup() {
    if [[ -n "${heartbeat_pid}" ]] && kill -0 "${heartbeat_pid}" > /dev/null 2>&1; then
        kill "${heartbeat_pid}" > /dev/null 2>&1 || true
    fi
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" > /dev/null 2>&1; then
            kill "${pid}" > /dev/null 2>&1 || true
        fi
    done
    wait || true
}

trap cleanup INT TERM EXIT

for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    port=$((start_port + worker_id))
    log_file=${log_dir}/worker_${host_ip}_worker_${worker_id}_port_${port}.log
    
    # Each worker uses 2 GPUs: worker_id*2 and worker_id*2+1
    gpu_start=$((worker_id * tensor_parallel_size))
    gpu_end=$((gpu_start + tensor_parallel_size - 1))
    cuda_devices="${gpu_start}"
    for ((g = gpu_start + 1; g <= gpu_end; g++)); do
        cuda_devices="${cuda_devices},${g}"
    done

    echo "Starting vLLM worker ip=${host_ip} worker=${worker_id} gpus=${cuda_devices} port=${port} log=${log_file}"
    CUDA_VISIBLE_DEVICES=${cuda_devices} python3 -m vllm.entrypoints.openai.api_server \
        --model "${model_path}" \
        --host "${host}" \
        --port "${port}" \
        --dtype "${dtype}" \
        --gpu-memory-utilization "${gpu_memory_utilization}" \
        --max-model-len "${max_model_len}" \
        --tensor-parallel-size "${tensor_parallel_size}" \
        "$@" > "${log_file}" 2>&1 &
    pids+=($!)
    ports+=("${port}")
done

echo "Workers launched:"
for ((gpu_id = 0; gpu_id < num_workers; gpu_id++)); do
    port=$((start_port + gpu_id))
    echo "  http://${host_ip}:${port}/v1"
done
echo "Heartbeat interval: ${heartbeat_interval}s"
print_heartbeat
heartbeat_loop &
heartbeat_pid=$!

wait "${pids[@]}"
