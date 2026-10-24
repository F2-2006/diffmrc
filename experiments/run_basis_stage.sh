#!/bin/bash
# One stage of the basis-set gradient validation; the stage name is $1.
# Relocatable: paths derive from this script's location; the interpreter
# is ${PYTHON:-python} so the caller supplies the environment.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
export DIFFMRC_SRC="$HERE/src"
export BSV_SYS="${BSV_SYS:-H2O_DZ22}"
export BSV_OUT="${BSV_OUT:-results/basis_set_${BSV_SYS}.json}"
export BSV_RELEASE="${BSV_RELEASE:-1}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export MKL_CBWR=COMPATIBLE
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
mkdir -p results logs
exec "${PYTHON:-python}" experiments/basis_set_gradient_validation.py "$1"
