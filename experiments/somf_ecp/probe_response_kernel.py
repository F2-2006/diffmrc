#!/usr/bin/env python
"""Check the density-response kernel of the two-electron SOMF term.

The value operator is linear in the reference density P:

    h2[P]_{pq} = sum_{rs} P_{rs} K[p,q,r,s]
    K[p,q,r,s] = g[p,q,r,s] - 3/2 g[p,r,s,q] - 3/2 g[s,q,p,r]

For a matrix element T = sum_{pq} D_{pq} h_{pq}[P] the density response
is

    dT_resp = sum_{rs} dP_{rs} M_{rs},
    M_{rs} = sum_{pq} D_{pq} K[p,q,r,s].

This probe checks whether M equals h2[D] (i.e. whether the kernel is
symmetric under the pair exchange (pq) <-> (rs)), for symmetric P and
antisymmetric transition densities D.  If it does, the response term can
reuse the value contraction; if not, an explicit response contraction is
required.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("MKL_CBWR", "COMPATIBLE")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from pyscf import gto

GEF = gto.M(atom=[("Ge", (0.0, 0.0, 0.0)), ("F", (0.5, 0.3, 2.4))],
            basis={"Ge": "lanl2dz", "F": "6-31g"},
            ecp={"Ge": "ecpds28mdfso"}, spin=1, unit="Bohr", verbose=0)


def maxabs(x):
    return float(np.max(np.abs(x))) if np.size(x) else 0.0


def kernel_tensor(mol):
    nao = mol.nao_nr()
    g = np.asarray(mol.intor("int2e_p1vxp1", aosym="s1"),
                   dtype=float).reshape(3, nao, nao, nao, nao)
    t = (g
         - 1.5 * g.transpose(0, 1, 3, 4, 2)
         - 1.5 * g.transpose(0, 4, 2, 1, 3))
    return g, t


def h2_from_tensor(t, p):
    return np.einsum("rs,kpqrs->kpq", p, t, optimize=True)


def response_from_tensor(t, d):
    return np.einsum("pq,kpqrs->krs", d, t, optimize=True)


def main():
    mol = GEF
    nao = mol.nao_nr()
    rng = np.random.default_rng(11)
    p = rng.standard_normal((nao, nao))
    p = 0.5 * (p + p.T)
    d = rng.standard_normal((nao, nao))
    d = 0.5 * (d - d.T)

    g, t = kernel_tensor(mol)
    h2 = h2_from_tensor(t, p)
    m = response_from_tensor(t, d)
    h2d = h2_from_tensor(t, d)

    print("kernel pair symmetry: max|K[pqrs]-K[rspq]| = %.3e (scale %.3e)"
          % (maxabs(t - t.transpose(0, 3, 4, 1, 2)), maxabs(t)))
    print("h2 antisym: max|h2+h2^T| = %.3e" % maxabs(h2 + h2.transpose(0, 2, 1)))
    print("M vs h2[D]: maxabs=%.3e rel=%.3e"
          % (maxabs(m - h2d), maxabs(m - h2d) / max(maxabs(h2d), 1e-300)))
    print("M vs -h2[D]: maxabs=%.3e"
          % maxabs(m + h2d))
    print("M vs h2[D]^T: maxabs=%.3e"
          % maxabs(m - h2d.transpose(0, 2, 1)))
    print("M antisym: max|M+M^T| = %.3e" % maxabs(m + m.transpose(0, 2, 1)))

    # Also check the naive direct route: h2_two_electron(mol, D) for the
    # antisymmetric D, which pyscf's direct_mapdm accepts as a matrix.
    from diffmrc.core import somf as dsomf
    h2d_direct = dsomf.somf_two_electron(mol, d, contract="direct")
    print("direct h2[D] vs explicit h2[D]: rel=%.3e"
          % (maxabs(h2d_direct - h2d) / max(maxabs(h2d), 1e-300)))
    print("density response via <dP, h2[D]> vs <dP, M>: both are the same "
          "quantity by construction")


if __name__ == "__main__":
    main()
