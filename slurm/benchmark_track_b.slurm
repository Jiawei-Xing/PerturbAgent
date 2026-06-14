#!/bin/bash
#SBATCH --job-name=bench_bsharp_parser_num
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=08:00:00
# Cluster-specific node pin (CSHL Elzar); delete or edit for your site:
##SBATCH --nodelist=bamgpu23,bamgpu24,bamgpu25
#SBATCH --output=bench_bsharp_parser_num.%j.out
#SBATCH --error=bench_bsharp_parser_num.%j.err

set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="${BIOREASONING_ENV:-$HOME/.conda/envs/bioreasoning}/bin:$PATH"  # override via BIOREASONING_ENV
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
module load cuda12.3/toolkit/12.3.2
export VLLM_USE_FLASHINFER_SAMPLER=0

# ── Sharp-judge arm: sharpened numeric DE judge (base-rate anchor +
# mechanistic-flag use + anti famous-regulator bias + spread). Keyword pubmed
# (RAG OFF) so this isolates the PROMPT effect. Deterministic sample
# (random_state=0) = identical 250 rows as benchmark_b_250 -> debate1_num is
# directly comparable to that run's 0.525. Only the numeric DE judge changes,
# so the three logprob conditions match baseline (harmless).
export MLGENX_SHARP_JUDGE=1

echo "[$(date)] node=$(hostname)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

# Dossier tools call mygene.info / STRING / NCBI E-utilities over HTTPS;
# degrade gracefully if offline. Probe once, non-fatally.
curl -sf --max-time 10 https://mygene.info/v3/query?q=symbol:Stat1 >/dev/null 2>&1 \
    && echo "[$(date)] outbound HTTPS OK (mygene reachable)" \
    || echo "[$(date)] WARNING: mygene unreachable -- dossier network tools will be empty."

# Preflight: serve-env torch is cu130; bail fast on an old-driver node.
uv run --extra serve python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('[preflight] torch CUDA OK', torch.version.cuda)" \
    || { echo "[$(date)] ERROR: torch CUDA preflight failed on $(hostname). Aborting."; exit 1; }

# ── Start vLLM with the logprobs JSON fix (logprob judges emit -inf) ──────
echo "[$(date)] starting vLLM serve (logprobs fix) ..."
uv run --extra serve python serve_with_logprobs_fix.py openai/gpt-oss-120b \
    --port 8001 --enforce-eager --no-enable-prefix-caching \
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
    if curl -sf http://localhost:8001/v1/models >/dev/null 2>&1; then READY=1; echo "[$(date)] server up after ~$((i*15))s"; break; fi
    sleep 15
done
[ "$READY" -ne 1 ] && { echo "[$(date)] server not ready; tail:"; tail -n 40 vllm_server.${SLURM_JOB_ID}.log; exit 1; }

echo "[$(date)] running Track B 250-row benchmark (debate1_num only, parser patch) ..."
uv run --extra serve python examples/benchmark_track_b_num.py \
    --api-base http://localhost:8001/v1 \
    --sample-size 250 \
    --out-dir outputs/benchmark_b_sharp_parser_num \
    --concurrency 16
RC=$?

echo "[$(date)] benchmark finished rc=$RC"
exit $RC
