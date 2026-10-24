#!/bin/bash
# Feasibility probe; relocatable and interpreter-agnostic (${PYTHON}).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd)
cd "$HERE"
export DIFFMRC_SRC="$HERE/src"
export BSV_BASIS="${BSV_BASIS:-cc-pvdz}"
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
export MKL_CBWR=COMPATIBLE
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
exec "${PYTHON:-python}" experiments/probe_basis_dz.py
