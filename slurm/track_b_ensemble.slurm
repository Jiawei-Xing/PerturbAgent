#!/bin/bash
#SBATCH --job-name=tb_ens
#SBATCH --partition=gpuq
#SBATCH --qos=bio_ai
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=23:00:00
#SBATCH --nodelist=bamgpu15,bamgpu19,bamgpu20,bamgpu26,bamgpu100
#SBATCH --output=tb_ens.%j.out
#SBATCH --error=tb_ens.%j.err

set -uo pipefail
cd "$(dirname "$0")/.."

export PATH="/grid/siepel/home/xing/.conda/envs/bioreasoning/bin:$PATH"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=INFO
module load cuda12.3/toolkit/12.3.2
export VLLM_USE_FLASHINFER_SAMPLER=0
export MLGENX_SHARP_JUDGE=1  # production config (matches shipped seed-42 submission)

S43=outputs/track_b_adversarial_sharp_seed43
S44=outputs/track_b_adversarial_sharp_seed44

echo "[$(date)] node=$(hostname)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

curl -sf --max-time 10 https://mygene.info/v3/query?q=symbol:Stat1 >/dev/null 2>&1 \
    && echo "[$(date)] outbound HTTPS OK (mygene reachable)" \
    || echo "[$(date)] WARNING: mygene unreachable -- dossier network tools will be empty."

uv run --extra serve python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('[preflight] torch CUDA OK', torch.version.cuda)" \
    || { echo "[$(date)] ERROR: torch CUDA preflight failed on $(hostname). Aborting."; exit 1; }

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

# ── Seed 43: full test run, gathers the (seed-independent) dossier once. ──────
echo "[$(date)] === SEED 43 ==="
uv run --extra serve python examples/track_b_adversarial.py \
    --api-base http://localhost:8000/v1 \
    --judge-mode numeric --rounds 1 --concurrency 16 --seed 43 \
    --output-dir $S43
RC43=$?
echo "[$(date)] seed43 finished rc=$RC43"

# ── Share seed-43's dossiers with seed-44 so we don't re-gather ~12k tool
# calls. Dossiers are pure tool output, independent of the sampling seed.
mkdir -p $S44
if [ -f "$S43/dossiers.json" ]; then
    cp "$S43/dossiers.json" "$S44/dossiers.json"
    echo "[$(date)] copied $S43/dossiers.json -> $S44/ (dossier reuse)"
fi

# ── Seed 44: reuses dossiers, only runs the seed-dependent debates. ──────────
echo "[$(date)] === SEED 44 ==="
uv run --extra serve python examples/track_b_adversarial.py \
    --api-base http://localhost:8000/v1 \
    --judge-mode numeric --rounds 1 --concurrency 16 --seed 44 \
    --output-dir $S44
RC44=$?
echo "[$(date)] seed44 finished rc=$RC44"

# ── Merge seed 42 (shipped) + 43 + 44 into the ensemble submission. ──────────
echo "[$(date)] === BUILD ENSEMBLE SUBMISSION ==="
uv run --extra serve python examples/build_ensemble_submission.py \
    --base-submission outputs/track_b_adversarial_sharp/submission.csv \
    --seed-cache outputs/track_b_adversarial_sharp/responses_cache.json \
    --seed-cache $S43/responses_cache.json \
    --seed-cache $S44/responses_cache.json \
    --out-dir outputs/track_b_adversarial_ens3
RCM=$?

echo "[$(date)] done. rc43=$RC43 rc44=$RC44 merge=$RCM"
[ $RC43 -eq 0 ] && [ $RC44 -eq 0 ] && [ $RCM -eq 0 ]
