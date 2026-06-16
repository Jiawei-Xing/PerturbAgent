#!/bin/bash
#SBATCH --job-name=gf_probe
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=gf_probe.%j.out
#SBATCH --error=gf_probe.%j.err
# Pin good-driver (>=580) nodes on submit:
#   sbatch --nodelist=bamgpu15,bamgpu18,bamgpu19,bamgpu23 slurm/geneformer_probe.slurm
# (serve-extra torch is cu130 -> needs driver >=580; bamgpu20/24 have 545 and fail.)

set -uo pipefail
# Slurm copies the batch script to a spool dir, so $0 is not the repo.
# SLURM_SUBMIT_DIR is where `sbatch` was invoked (the repo root); fall back to
# $0 for direct (non-Slurm) execution.
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

# uv lives in the bioreasoning conda env; Slurm doesn't inherit the login PATH.
export PATH="${BIOREASONING_ENV:-/grid/siepel/home/xing/.conda/envs/bioreasoning}/bin:$PATH"
export HF_HUB_OFFLINE=1

echo "[$(date)] node=$(hostname)"; nvidia-smi --query-gpu=driver_version,name --format=csv,noheader

uv run --extra serve python -c "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); print('[preflight] torch CUDA OK', torch.version.cuda)" \
    || { echo "[$(date)] ERROR: torch CUDA preflight failed on $(hostname). Aborting."; exit 1; }

echo "[$(date)] running Geneformer in-silico perturbation probe ..."
uv run --extra serve python examples/geneformer_probe.py \
    --sample outputs/benchmark_b_250/sample.csv \
    --train data/train.csv \
    --predict data/test.csv \
    --batch 8 \
    --out outputs/geneformer_probe
RC=$?
echo "[$(date)] probe finished rc=$RC"
exit $RC
