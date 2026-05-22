#!/bin/bash
#SBATCH --job-name=irp_fof_extract
#SBATCH --output=logs/irp_fof_extract_%j.out
#SBATCH --error=logs/irp_fof_extract_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1

set -euo pipefail

cd "$HOME/irp-nny124"

mkdir -p outputs logs

# Load Python/h5py environment if needed.
# Try common Imperial/HPC module patterns, but do not fail if modules are unavailable.
module load anaconda3 2>/dev/null || true
module load hdf5 2>/dev/null || true

python scripts/extract_fof_outcomes.py \
  --data-dir "$EPHEMERAL/martian_moons_data" \
  --outputs-dir outputs \
  --schema-samples 5 \
  --min-particles 20 \
  --exclude-group-id -1
