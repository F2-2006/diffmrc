"""pyscf state-averaged CASSCF energy and gradient for the same system."""
import numpy as np
from pyscf import gto, mcscf, scf
from pyscf.x2c import sfx2c1e

GEOM = [('O', (0.0, 0.0, 0.0)),
        ('H', (0.0, -0.757, 0.587)),
        ('H', (0.0, 0.757, 0.587))]
AUX = 'def2-universal-jkfit'
NCORE, NACT, NELE = 4, 2, 2


def main():
    mol = gto.M(atom=GEOM, basis='6-31g', unit='Angstrom', verbose=0)
    mf = scf.RHF(mol).density_fit(auxbasis=AUX)
    mf = sfx2c1e.sfx2c1e(mf)
    mf.conv_tol = 1e-12
    mf.kernel()
    mc = mcscf.CASSCF(mf, NACT, NELE)
    mc.state_average_([0.5, 0.5])
    mc.conv_tol = 1e-12
    mc.max_cycle = 200
    mc.kernel()
    print('e_tot', mc.e_tot, flush=True)
    print('e_states', np.asarray(mc.e_states), flush=True)
    print('weights', mc.fcisolver.weights, flush=True)
    try:
        g = mc.nuc_grad_method().kernel()
        print('sa gradient:\n', np.array2string(np.asarray(g),
                                                 precision=10), flush=True)
    except Exception as ex:
        print('grad error', type(ex).__name__, ex, flush=True)
    # state-specific attempt: plain CASSCF with nroots/weights set
    from pyscf.mcscf import StateAverageMCSCFSolver
    print('is SA solver', isinstance(mc, StateAverageMCSCFSolver), flush=True)
    mc2 = mcscf.CASSCF(mf, NACT, NELE)
    mc2.fcisolver.nroots = 2
    mc2.fcisolver.weights = [0.5, 0.5]
    mc2.conv_tol = 1e-12
    mc2.max_cycle = 200
    mc2.kernel()
    print('plain-object e_tot', mc2.e_tot, 'e_states',
          np.asarray(mc2.e_states), flush=True)
    print('is SA solver 2', isinstance(mc2, StateAverageMCSCFSolver),
          'grad state', mc2.nuc_grad_method().state, flush=True)
    for st in (0, 1):
        try:
            g2 = mc2.nuc_grad_method().kernel(state=st)
            print('state', st, 'grad:', flush=True)
            print(np.array2string(np.asarray(g2), precision=10), flush=True)
        except Exception as ex:
            print('state', st, 'error', type(ex).__name__, ex, flush=True)


if __name__ == '__main__':
    main()
