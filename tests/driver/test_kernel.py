"""Assembling the end-to-end path out of the frozen kernel's pieces."""
import numpy as np
import pytest

from diffmrc.driver.kernel import ActiveSpace, partition_from_occupations


def test_partition_splits_around_the_active_set():
    occ = np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0])
    space = partition_from_occupations(occ, [2, 3])
    assert space.core == [0, 1]
    assert space.active == [2, 3]
    assert space.virtual == [4, 5]
    assert space.n_elec == 2


def test_partition_reads_occupations_not_index_order():
    """The failure this exists to prevent.

    A multi-solution SCF can leave the singly occupied orbitals in a
    block the aufbau order does not predict. Assuming 'the first ncore
    orbitals are doubly occupied, the next nact are active' then selects
    empty orbitals: this project measured a CASCI 18.67 Ha above the
    ROHF it started from, which is impossible for an exact solver.
    """
    occ = np.array([2.0, 0.0, 2.0, 1.0, 0.0, 0.0])
    space = partition_from_occupations(occ, [1, 3])
    assert space.core == [0, 2]
    assert space.active == [1, 3]
    assert space.n_elec == 1


def test_partition_counts_electrons_from_the_occupations():
    occ = np.array([2.0, 2.0, 2.0, 0.0])
    space = partition_from_occupations(occ, [1, 2])
    assert space.n_elec == 4
    assert space.ncas == 2


def test_partition_rejects_an_index_outside_the_reference():
    occ = np.array([2.0, 0.0])
    with pytest.raises(ValueError):
        partition_from_occupations(occ, [2])


def test_partition_deduplicates_the_active_set():
    occ = np.array([2.0, 1.0, 0.0])
    space = partition_from_occupations(occ, [1, 1])
    assert space.active == [1]


class _FakeMol:
    """A stand-in for a pyscfad Mole: fixed integrals, no SCF.

    Three orbitals with two electrons in the first two: the third is the
    virtual that gives the rotation parameters something to do. The
    integrals are a model, not a physical molecule. Tests built on it
    assert *properties* -- parameter counts, the variational bound,
    stationarity -- and never an absolute energy recalled from a paper;
    the absolute check runs on the server against the pinned reference
    in tests/core/test_frozen_reference.py.
    """

    def __init__(self):
        self._h = np.array([[0.0, 0.0, 0.0],
                            [0.0, -0.2, 0.0],
                            [0.0, 0.0, 0.3]])
        self._eri = np.zeros((3, 3, 3, 3))
        for i, v in enumerate((0.80, 0.60, 0.40)):
            self._eri[i, i, i, i] = v
        self._eri[0, 0, 1, 1] = self._eri[1, 1, 0, 0] = 0.55
        self._eri[0, 0, 2, 2] = self._eri[2, 2, 0, 0] = 0.35
        self._eri[1, 1, 2, 2] = self._eri[2, 2, 1, 1] = 0.30
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._eri[i, j, i, j] = 0.10
                    self._eri[i, j, j, i] = 0.10

    def nao_nr(self):
        return 3

    def intor(self, name, **kwargs):
        if name == 'int2e':
            return self._eri
        if name == 'int1e_kin':
            return self._h
        if name == 'int1e_nuc':
            return np.zeros((3, 3))     # the one-electron part rides in kin
        raise ValueError(f'unexpected integral {name!r}')

    def energy_nuc(self):
        return 0.0


def test_energy_function_feeds_the_kernel_the_transformed_integrals():
    """At theta = 0 the rotated energy is the CASCI energy itself.

    The test transforms the integrals with explicit loops -- a different
    spelling from the implementation's einsums -- so an index-order slip
    in either shows up as a disagreement.
    """
    # The kernel's CASCI imports pyscfad, so the energy path is
    # exercised where pyscfad is installed; the properties asserted here
    # are the ones that do not need a real molecule.
    pytest.importorskip('pyscfad')
    from diffmrc.core.casci import cs_casci_blocked
    from diffmrc.driver.kernel import energy_function

    mol = _FakeMol()
    C0 = np.eye(3)
    space = ActiveSpace(core=[], active=[0, 1], virtual=[2], n_elec=2)
    E_theta, npar = energy_function(mol, C0, space)
    assert npar == 2                      # orbitals {0,1} against {2}
    h1 = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            h1[i, j] = mol._h[i, j]
    e1 = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    e1[i, j, k, l] = mol._eri[i, k, j, l]
    direct = float(cs_casci_blocked(h1, e1, ncore=0, nact=2, nele_act=2)[0])
    assert float(E_theta(np.zeros(npar))) == pytest.approx(direct, abs=1e-12)


def test_the_search_reaches_a_lower_energy_than_the_reference():
    """The variational bound: the optimized point cannot sit above the
    CASCI it started from."""
    pytest.importorskip('pyscfad')
    from diffmrc.driver.kernel import energy_function, search_point

    mol = _FakeMol()
    C0 = np.eye(3)
    space = ActiveSpace(core=[], active=[0, 1], virtual=[2], n_elec=2)
    E_theta, npar = energy_function(mol, C0, space)
    point = search_point(E_theta, npar)
    assert point.is_stationary
    assert point.energy <= float(E_theta(np.zeros(npar))) + 1e-12


def _fake_point(theta, energy):
    """An OrbitalPoint with prescribed numbers, for the faked convergence."""
    from diffmrc.driver.orbitals import OrbitalPoint

    class _Refinement:
        is_stationary = True
        stop_reason = None
        target = 1e-11
        n_accepted = 1

    return OrbitalPoint(theta=np.asarray(theta), energy=energy,
                        grad_norm=1e-12, optimizer_grad_norm=1e-12,
                        optimizer_message='faked', optimizer_nit=1,
                        optimizer_reached=True, refinement=_Refinement())


def test_multistart_keeps_the_variationally_lowest_point(monkeypatch):
    """The selection rule, isolated from any optimizer.

    A faked convergence returns a prescribed energy per start, so the
    test is about which point `search_point` keeps and which starts it
    asks for, not about a landscape.
    """
    from diffmrc.driver import kernel

    seen = []

    def _fake_converge(E_theta, npar, theta0=None, **kwargs):
        start = np.asarray(theta0, dtype=float).copy()
        seen.append(start)
        return _fake_point(start, -1.0 - float(np.sum(start ** 2)))

    monkeypatch.setattr(kernel, 'converge_orbital_point', _fake_converge)
    best = kernel.search_point(lambda t: 0.0, 3, tries=4, seed=5)
    assert np.allclose(seen[0], np.zeros(3))          # first start is theta=0
    assert len(seen) == 4
    # The kept point is the one whose start had the largest norm, i.e.
    # the lowest faked energy.
    assert best.energy == min(-1.0 - float(np.sum(s ** 2)) for s in seen)


def test_multistart_is_reproducible_for_a_given_seed(monkeypatch):
    """Seeded starts: the same call asks for the same points twice."""
    from diffmrc.driver import kernel

    starts = []

    def _fake_converge(E_theta, npar, theta0=None, **kwargs):
        start = np.asarray(theta0, dtype=float).copy()
        starts.append(start)
        return _fake_point(start, -float(np.sum(start ** 2)))

    monkeypatch.setattr(kernel, 'converge_orbital_point', _fake_converge)
    kernel.search_point(lambda t: 0.0, 2, tries=3, seed=1)
    first = list(starts)
    del starts[:]
    kernel.search_point(lambda t: 0.0, 2, tries=3, seed=1)
    assert len(first) == len(starts) == 3
    for a, b in zip(first, starts):
        assert np.allclose(a, b)


def test_select_active_space_honours_a_manual_choice():
    from diffmrc.driver.kernel import select_active_space

    mol = _FakeMol()
    occ = np.array([2.0, 0.0, 0.0])
    C = np.eye(3)
    active, evidence = select_active_space(mol, C, occ, ['H', 'H'],
                                           manual=[0])
    assert active == [0]
    assert any('manual' in e for e in evidence)


def test_select_active_space_uses_the_rules_when_not_given_manually():
    from diffmrc.driver.kernel import select_active_space

    mol = _FakeMol()
    occ = np.array([2.0, 0.0, 0.0])
    C = np.eye(3)
    active, evidence = select_active_space(mol, C, occ, ['H', 'H'])
    # Either the rules propose something and say why, or they refuse and
    # say why; both are answers. A silent empty result is the failure.
    assert evidence
    if active is not None:
        assert all(isinstance(i, int) for i in active)


def test_scf_reference_refuses_without_the_numerical_kernel():
    """Without pyscfad the call must refuse, not crash obscurely.

    Where pyscfad *is* installed the same call returns a molecule and a
    converged mean field, which is what the server validation exercises;
    both outcomes are asserted here so the test runs in either
    environment.
    """
    from diffmrc.driver.kernel import scf_reference

    try:
        import pyscfad  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            scf_reference('H 0 0 0; H 0 0 0.74', 0, 0, '6-31g')
    else:
        mol, mf = scf_reference('H 0 0 0; H 0 0 0.74', 0, 0, '6-31g')
        assert mol.nao_nr() > 0
        assert bool(mf.converged)
