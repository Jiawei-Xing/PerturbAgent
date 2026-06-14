#!/bin/bash
#SBATCH --job-name=tb_adv_sharp
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --nodelist=bamgpu15,bamgpu18,bamgpu19
#SBATCH --output=tb_adv_sharp.%j.out
#SBATCH --error=tb_adv_sharp.%j.err

set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="/grid/siepel/home/xing/.conda/envs/bioreasoning/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
module load cuda12.3/toolkit/12.3.2
export VLLM_USE_FLASHINFER_SAMPLER=0
export MLGENX_SHARP_JUDGE=1  # production config: sharpened DE judge (benchmark-confirmed DE 0.511->0.576)

echo "[$(date)] node=$(hostname)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

# Dossier tools call mygene.info / STRING / PubMed over HTTPS. Over ~1800 rows
# this is thousands of external calls; failures degrade gracefully (error
# strings) and dossiers are cached/resumable. Probe once, non-fatally.
curl -sf --max-time 10 https://mygene.info/v3/query?q=symbol:Stat1 >/dev/null 2>&1 \
    && echo "[$(date)] outbound HTTPS OK (mygene reachable)" \
    || echo "[$(date)] WARNING: mygene unreachable -- dossier network tools will be empty."

# Preflight: serve-env torch is cu130; bail fast on an old-driver node.
uv run --extra serve python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('[preflight] torch CUDA OK', torch.version.cuda)" \
    || { echo "[$(date)] ERROR: torch CUDA preflight failed on $(hostname). Aborting."; exit 1; }

# ── Start vLLM (numeric judges need no logprobs, but the fix wrapper is
# harmless and keeps parity with the benchmark serving setup) ─────────────
echo "[$(date)] starting vLLM serve ..."
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

# ── Full test run: adversarial debate agent on all of data/test.csv with the
# benchmark-winning config (1 round, numeric judges). Writes submission.csv +
# submission_track_b.zip + prompt.txt + tools/ to outputs/track_b_adversarial/.
# Per-row cache makes the run resumable on requeue.
echo "[$(date)] running Track B adversarial agent on test set ..."
uv run --extra serve python examples/track_b_adversarial.py \
    --api-base http://localhost:8000/v1 \
    --judge-mode numeric \
    --rounds 1 \
    --concurrency 16 \
    --output-dir outputs/track_b_adversarial_sharp
RC=$?

echo "[$(date)] agent finished rc=$RC"
exit $RC
