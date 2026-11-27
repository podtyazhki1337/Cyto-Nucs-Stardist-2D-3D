#!/usr/bin/env bash
# Runs a command of the CytoNucs image (see docker/cytonucs, or "docker/run.sh help") on this machine's GPUs, as the
# current user. Mounted from the current folder: data/ (datasets), runs/ (run folders), models/ (pretrained models),
# where the configurations of the paper expect them, and the folder itself at /work.
#
#   docker/run.sh segment helacytonuc_e7_clean my_images my_labels
#   docker/run.sh python train_cytonucs.py configs/helacytonuc_e7_clean.json
#   docker/run.sh bash
#
# Environment: CYTONUCS_IMAGE (default cytonucs), CYTONUCS_GPUS (default all; e.g. device=1, or none for the CPU),
# CYTONUCS_DATA, CYTONUCS_RUNS, CYTONUCS_MODELS (other host folders for data/, runs/, models/).
set -euo pipefail

image=${CYTONUCS_IMAGE:-cytonucs}
data=${CYTONUCS_DATA:-$PWD/data}
runs=${CYTONUCS_RUNS:-$PWD/runs}
models=${CYTONUCS_MODELS:-$PWD/models}
gpus=${CYTONUCS_GPUS:-all}

mkdir -p "$runs" "$models"
args=(--rm --user "$(id -u):$(id -g)" -e CYTONUCS_CWD=/work -v "$PWD:/work"
      -v "$(realpath "$runs"):/opt/cytonucs/runs" -v "$(realpath "$models"):/opt/cytonucs/models")
if [ -d "$data" ]; then
    args+=(-v "$(realpath "$data"):/opt/cytonucs/data")
fi
if [ "$gpus" != none ]; then
    args+=(--gpus "$gpus")
fi
if [ -t 0 ] && [ -t 1 ]; then
    args+=(-it)
fi
exec docker run "${args[@]}" "$image" "$@"
