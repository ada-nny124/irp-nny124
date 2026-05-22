#!/bin/bash
#SBATCH --job-name=irp_fof_extract
#SBATCH --output=logs/irp_fof_extract_%j.out
#SBATCH --error=logs/irp_fof_extract_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/irp-nny124}"
DATA_DIR="${DATA_DIR:-$EPHEMERAL/martian_moons_data}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$REPO_DIR/outputs}"
SCHEMA_SAMPLES="${SCHEMA_SAMPLES:-3}"
MIN_PARTICLES="${MIN_PARTICLES:-20}"
EXCLUDE_GROUP_ID="${EXCLUDE_GROUP_ID:--1}"

cd "$REPO_DIR"
mkdir -p "$OUTPUTS_DIR" logs

module load tools/prod h5py/3.12.1-foss-2024a

python3 scripts/extract_fof_outcomes.py \
  --data-dir "$DATA_DIR" \
  --outputs-dir "$OUTPUTS_DIR" \
  --schema-samples "$SCHEMA_SAMPLES" \
  --min-particles "$MIN_PARTICLES" \
  --exclude-group-id "$EXCLUDE_GROUP_ID"
