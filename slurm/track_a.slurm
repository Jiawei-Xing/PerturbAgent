#!/bin/bash
#SBATCH --job-name=track_a_final
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --nodelist=bamgpu15,bamgpu18,bamgpu19
#SBATCH --output=track_a_final.%j.out
#SBATCH --error=track_a_final.%j.err

set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="/grid/siepel/home/xing/.conda/envs/bioreasoning/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
module load cuda12.3/toolkit/12.3.2
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "[$(date)] node=$(hostname)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

# Preflight: serve-env torch is cu130; abort fast on an old-driver node.
uv run --extra serve python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('[preflight] torch CUDA OK', torch.version.cuda)" \
    || { echo "[$(date)] ERROR: torch CUDA preflight failed on $(hostname). Aborting."; exit 1; }

# Serve with the logprobs JSON fix so -inf logprobs serialize.
echo "[$(date)] starting vLLM serve (logprobs fix) ..."
uv run --extra serve python serve_with_logprobs_fix.py openai/gpt-oss-120b \
    --port 8000 --enforce-eager --no-enable-prefix-caching \
    --tensor-parallel-size 1 --gpu-memory-utilization 0.95 \
    > vllm_server.${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
cleanup() { echo "[$(date)] cleaning up vLLM (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

echo "[$(date)] waiting for server ..."
READY=0
for i in $(seq 1 120); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM died during startup. Tail:"; tail -n 40 vllm_server.${SLURM_JOB_ID}.log; exit 1
    fi
    if curl -sf http://localhost:8000/v1/models >/dev/null 2>&1; then READY=1; echo "[$(date)] server up after ~$((i*15))s"; break; fi
    sleep 15
done
[ "$READY" -ne 1 ] && { echo "[$(date)] server not ready; tail:"; tail -n 40 vllm_server.${SLURM_JOB_ID}.log; exit 1; }

# Final Track A submission: v2 prompt + logprob extraction + medium effort
# (the offline-benchmark winner). Fresh output dir so no stale cache replay.
echo "[$(date)] running Track A final (logprobs, v2, medium) ..."
uv run --extra serve python examples/track_a_logprobs.py \
    --api-base http://localhost:8000/v1 \
    --prompt-template examples/prompt_template_v2.txt \
    --reasoning-effort medium \
    --output-dir outputs/track_a/v2_logprobs \
    --concurrency 20
RC=$?

echo "[$(date)] Track A final finished rc=$RC"
ls -la outputs/track_a/v2_logprobs 2>/dev/null
exit $RC
