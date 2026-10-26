"""Valence-level spin-orbit mean field (SOMF) for effective core potentials.

WHY THIS MODULE EXISTS

``diffmrc.core.somf`` implements the all-electron SOMF operator: a
one-electron Breit-Pauli term with the bare nuclear charges plus the
``J - 3/2 K`` mean field of the full electron density.  With an effective
core potential (ECP) the core electrons are absent from the calculation,
and the spin-orbit part of the ECP (``ECPso``) replaces their contribution
to the spin-orbit operator.  Adding the all-electron formula unchanged
would use the ECP effective charge in ``int1e_pnucxp`` and a valence-only
density, and would double count or omit core contributions.  This module
defines and implements a valence-level operator instead.

DEFINITION USED HERE

For a molecule in which some centers carry a spin-orbit ECP:

    h^SOMF_ECP,k = (alpha^2/2) * [ h^nuc,k + h^2e,k[P] ] - V^ECP,k

with

    h^nuc,k = sum_{A not SO-ECP} Z_eff,A (r_iA x p_i)_k / r_iA^3
              (the ``int1e_pnucxp`` convention; Z_eff,A is the ECP
              effective charge on scalar-ECP centers and the full
              charge on all-electron centers),

    h^2e,k[P] = sum_{rs} P_rs [ (pq|g_k|rs) - 3/2 (pr|g_k|sq)
                               - 3/2 (sq|g_k|pr) ],

    V^ECP,k = the three-component ``ECPso`` integral,

and ``P`` is the spin-summed density of the electrons explicitly present
in the calculation, ``trace(P S) = mol.nelectron``.  The global factor
``alpha^2/2`` is carried by the Breit-Pauli terms only; ``ECPso`` is
already an effective operator in energy units.  The sign of the ECP term
is the pyscf GHF convention (``H = -i sum_k V_k s_k``), while the
Breit-Pauli terms use the pyscf ZFS convention (``H = +i sum_k h_k s_k``);
in the common ``+i`` convention the returned operator therefore enters
``H_SO = i sum_k h_k s_k`` and the ECP term is subtracted.

WHY THE ECP TERM REPLACES THE NUCLEAR TERM ON ITS CENTER

A shape-consistent or energy-consistent spin-orbit ECP is fitted to
reproduce the all-electron spin-orbit splitting of the valence
configuration, so it already contains the nuclear one-electron term and
the core-valence spin-other-orbit screening for that center.  On YbF the
raw ``int1e_pnucxp`` norm on Yb is about 1.2e4 while the ECPso norm is
17; adding both would be dominated by an operator that the ECP was built
to replace.  On centers without a spin-orbit ECP the nuclear term is the
only available one-electron spin-orbit operator, and it is kept (with
``Z_eff`` from the atom table).

WHAT IS NOT UNIQUE

There is no unique first-principles decomposition of an ECP into one- and
two-electron spin-orbit parts.  The ECP is fitted to atomic reference
states, so part of the valence-valence mean field is already folded into
its parameters.  Adding the explicit valence ``J - 3/2 K`` term is
therefore an approximation with a double-counting error that cannot be
removed without the ECP generation data.  The module exposes the pieces
independently so that the error can be bounded by comparing variants; it
does not choose a "physical" amount of valence mean field on its own.

THE VARIANT FAMILY

:func:`somf_variants` returns the operator for every decomposition that
the ECP data alone cannot distinguish, all in the same ``+i``
convention:

* ``ecp``: ``ECPso`` only, the operator fixed by the ECP fit;
* ``nuclear_non_ecp``, ``breit_pauli``: the Breit-Pauli terms without
  ``ECPso``;
* ``valence``: the module default, ``ECPso`` plus the full valence mean
  field;
* ``one_center``: the mean-field kernel and density truncated to the
  spin-orbit ECP centers (AMFI-like);
* ``atomic_subtracted``: the full mean field minus its on-site block,
  i.e. the variant that assumes the ECP already contains the free-atom
  valence mean field;
* ``point_charge``: the scalar-ECP point-charge model, effective-charge
  nuclear SOC with no ``ECPso``;
* ``double``: ``ECPso`` plus the nuclear SOC on every center and the
  full mean field, a diagnostic upper view of the double counting.

The spread of these variants on a molecule is the quantitative statement
of the decomposition ambiguity; none of them is promoted to an absolute
spin-orbit operator.

DENSITY RESPONSE

The two-electron kernel ``K[p,q,r,s]`` is not symmetric under the pair
exchange ``(pq) <-> (rs)``.  Consequently the derivative of
``tr(D h2[P])`` with respect to ``P`` is *not* ``h2[D]``; it is a separate
contraction,

    M_rs = sum_{pq} D_pq K[p,q,r,s],

implemented in :func:`somf_density_response`.  A gradient of an SOC
matrix element must use ``M`` (or the equivalent bilinear form) and not
the value contraction.  The value layer here is numpy; orbital and CI
response are the caller's responsibility.
"""
from __future__ import annotations

import numpy as np

from .somf import (DEFAULT_MEMORY_LIMIT, SOMF_ALPHA2_HALF, one_electron_soc,
                   somf_two_electron)

__all__ = [
    "SOMF_ALPHA2_HALF",
    "ecp_soc",
    "so_ecp_centers",
    "nuclear_centers",
    "nuclear_soc",
    "valence_somf_matrix",
    "somf_variants",
    "somf_density_response",
    "somf_definition",
]

# pyscf atom-table slot constants (pyscf.gto.mole); duplicated here to keep
# the dependency lazy and the values explicit.
_ATOM_OF = 0
_SO_TYPE_OF = 4


def _atom_table(mol):
    """Return ``(atom_of, so_type)`` arrays from the ECP basis table."""
    ecpbas = np.asarray(getattr(mol, "_ecpbas", np.zeros((0, 8))),
                        dtype=int)
    if ecpbas.size == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    return ecpbas[:, _ATOM_OF], ecpbas[:, _SO_TYPE_OF]


def so_ecp_centers(mol):
    """Atom indices whose ECP carries a spin-orbit component.

    Returns
    -------
    list of int
        Sorted unique atom indices.  Empty when no ECP has a spin-orbit
        part.
    """
    atom_of, so_type = _atom_table(mol)
    if atom_of.size == 0:
        return []
    return sorted(int(x) for x in np.unique(atom_of[so_type == 1]))


def nuclear_centers(mol):
    """Atom indices without a spin-orbit ECP (nuclear SOC is kept).

    This includes all-electron centers and centers with a scalar ECP
    only.  On a scalar-ECP center the atom table carries ``Z_eff``, so
    the nuclear term is the effective-charge operator; that is the only
    one-electron spin-orbit operator available there.
    """
    so = set(so_ecp_centers(mol))
    return [ia for ia in range(mol.natm) if ia not in so]


def _center_ao_mask(mol, atoms):
    """Boolean mask of the AO functions centered on ``atoms``."""
    mask = np.zeros(mol.nao_nr(), dtype=bool)
    slices = mol.aoslice_by_atom()
    for ia in atoms:
        p0, p1 = int(slices[ia, 2]), int(slices[ia, 3])
        mask[p0:p1] = True
    return mask


def ecp_soc(mol):
    """Three-component ``ECPso`` integral, shape ``(3, nao, nao)``.

    The integral is real and antisymmetric per component and carries no
    ``alpha^2/2`` factor.  It is the effective one-electron spin-orbit
    operator of the ECP; the pyscf GHF convention is
    ``H = -i sum_k V_k s_k``.
    """
    if not mol.has_ecp_soc():
        raise ValueError("molecule has no spin-orbit ECP; use the "
                         "all-electron somf module instead")
    return np.asarray(mol.intor("ECPso"), dtype=float)


def nuclear_soc(mol, atoms=None):
    """Nuclear one-electron SOC in the ``int1e_pnucxp`` convention.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    atoms : sequence of int, optional
        Atom indices to include.  ``None`` includes every center and
        reproduces ``mol.intor_asymmetric('int1e_pnucxp', 3)`` to roundoff.

    Returns
    -------
    ndarray, shape (3, nao, nao)
        ``-sum_A Z_A int1e_prinvxp(A)``, real and antisymmetric.  No
        ``alpha^2/2`` factor.

    Notes
    -----
    ``int1e_pnucxp`` uses the atom-table charge, which on an ECP center
    is ``Z - N_core``.  The per-center construction here makes that
    convention explicit and allows the ECP centers to be excluded.
    """
    nao = mol.nao_nr()
    out = np.zeros((3, nao, nao), dtype=float)
    if atoms is None:
        atoms = range(mol.natm)
    for ia in atoms:
        ia = int(ia)
        with mol.with_rinv_at_nucleus(ia):
            v = np.asarray(mol.intor("int1e_prinvxp", comp=3), dtype=float)
        out -= float(mol.atom_charge(ia)) * v
    return out


def _two_electron_truncated(mol, P, mask, contract="auto",
                            memory_limit=None):
    """``J - 3/2 K`` with all four integral indices on the masked centers.

    The density is restricted to the selected AO block and the resulting
    operator is masked on its own AO pair, so the Coulomb and exchange
    contractions only use pairs that lie on a selected center.  With one
    selected center this is the one-center (AMFI-like) truncation of the
    spin-orbit mean field.
    """
    nao = mol.nao_nr()
    if not np.any(mask):
        return np.zeros((3, nao, nao), dtype=float)
    pm = P * np.outer(mask, mask)
    if contract == "auto":
        limit = DEFAULT_MEMORY_LIMIT if memory_limit is None else memory_limit
        need = 8.0 * 3 * nao ** 4
        contract = "explicit" if need <= limit else "direct"
    if contract not in ("explicit", "direct"):
        raise ValueError(f"unknown contract route: {contract!r}")
    if contract == "explicit":
        g = np.asarray(mol.intor("int2e_p1vxp1", aosym="s1"), dtype=float)
        g = g.reshape(3, nao, nao, nao, nao)
        j = np.einsum("rs,kpqrs->kpq", pm, g, optimize=True)
        k1 = np.einsum("rs,kprsq->kpq", pm, g, optimize=True)
        k2 = np.einsum("rs,ksqpr->kpq", pm, g, optimize=True)
        h = j - 1.5 * (k1 + k2)
    else:
        from pyscf.prop.gtensor.uhf import get_jk

        vj, vk = get_jk(mol, (0.5 * pm, 0.5 * pm))
        h = (vj[0] + vj[1]) - 1.5 * (vk[0] + vk[1])
    return h * np.outer(mask, mask)[None, :, :]


def valence_somf_matrix(mol, P, nuclear="auto", ecp_term=True,
                        two_electron="full", onsite="keep",
                        contract="auto", memory_limit=None,
                        convention="i", check=False,
                        one_electron="ecpso"):
    """Valence-level SOMF SOC operator, shape ``(3, nao, nao)``.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    P : array_like, shape (nao, nao)
        Spin-summed density of the explicitly treated electrons,
        ``trace(P S) = mol.nelectron``.
    nuclear : {'auto', 'all', 'none'}
        ``'auto'`` (default) keeps the nuclear SOC only on centers
        without a spin-orbit ECP.  ``'all'`` keeps it on every center
        (uses the effective charge on ECP centers; diagnostic only, it
        double counts).  ``'none'`` omits it.
    ecp_term : bool
        Include the ``ECPso`` operator.  ``False`` is diagnostic.
    two_electron : {'full', 'one_center', 'none'}
        ``'full'`` is the four-center ``J - 3/2 K`` mean field.
        ``'one_center'`` truncates the kernel and the density to the AO
        blocks of the spin-orbit ECP centers (an AMFI-like
        approximation).  ``'none'`` omits the term.
    onsite : {'keep', 'subtract'}
        ``'subtract'`` removes the spin-orbit ECP centers' on-site block
        from the density that generates the mean field.  That block is
        the free-atom valence density, which the ECP fit may already
        contain; the mode therefore removes the on-site part of the
        double counting and keeps only the inter-site (bonding) change.
    contract : {'auto', 'explicit', 'direct'}
        Forwarded to the two-electron contraction.
    memory_limit : float, optional
        Byte budget for the explicit kernel in ``'auto'``.
    convention : {'i', 'raw'}
        ``'i'`` (default) returns the operator in the ``+i`` convention
        ``H_SO = i sum_k h_k s_k``, i.e. the Breit-Pauli terms carry
        ``alpha^2/2`` and the ECP term is subtracted.  ``'raw'`` returns
        the unscaled components added with their native signs; it is a
        diagnostic view and a caller using it must apply the two
        different prefactors itself.
    check : bool
        Validate ``trace(P S)`` and antisymmetry.
    one_electron : {'ecpso', 'point_charge'}
        ``'ecpso'`` uses the ECP spin-orbit operator on the SO-ECP
        centers.  ``'point_charge'`` instead uses the effective-charge
        point-nucleus operator on every center and omits ``ECPso``; this
        is the scalar-ECP point-charge model and is a diagnostic view,
        not a physical decomposition.  It overrides ``nuclear`` and
        ``ecp_term``.

    Returns
    -------
    ndarray, shape (3, nao, nao)
        Real; antisymmetric when the density is symmetric and the
        operator is assembled consistently.
    """
    if convention not in ("i", "raw"):
        raise ValueError(f"unknown convention: {convention!r}")
    if nuclear not in ("auto", "all", "none"):
        raise ValueError(f"unknown nuclear mode: {nuclear!r}")
    if two_electron not in ("full", "one_center", "none"):
        raise ValueError(f"unknown two-electron mode: {two_electron!r}")
    if onsite not in ("keep", "subtract"):
        raise ValueError(f"unknown onsite mode: {onsite!r}")
    if one_electron not in ("ecpso", "point_charge"):
        raise ValueError(f"unknown one-electron mode: {one_electron!r}")

    nao = mol.nao_nr()
    P = np.asarray(P, dtype=float)
    if P.shape != (nao, nao):
        raise ValueError(f"P must have shape ({nao}, {nao}), got {P.shape}")
    if check:
        s = mol.intor("int1e_ovlp")
        nelec = float(np.einsum("pq,pq->", P, s))
        if abs(nelec - mol.nelectron) > 1e-6:
            raise ValueError(
                f"trace(P S) = {nelec:.9f} differs from mol.nelectron "
                f"= {mol.nelectron}")

    if one_electron == "point_charge":
        nuc_atoms = list(range(mol.natm))
        use_ecp = False
    else:
        use_ecp = bool(ecp_term)
        if nuclear == "all":
            nuc_atoms = list(range(mol.natm))
        elif nuclear == "none":
            nuc_atoms = []
        else:
            nuc_atoms = nuclear_centers(mol)

    h_bp = np.zeros((3, nao, nao), dtype=float)
    if nuc_atoms:
        h_bp += nuclear_soc(mol, nuc_atoms)

    if two_electron != "none":
        p_eff = P
        if onsite == "subtract":
            so_mask = _center_ao_mask(mol, so_ecp_centers(mol))
            if not np.any(so_mask):
                raise ValueError("onsite='subtract' requires an SO-ECP "
                                 "center")
            p_eff = P - P * np.outer(so_mask, so_mask)
        if two_electron == "one_center":
            mask = _center_ao_mask(mol, so_ecp_centers(mol))
            if not np.any(mask):
                raise ValueError("two_electron='one_center' requires an "
                                 "SO-ECP center")
            h_bp += _two_electron_truncated(
                mol, p_eff, mask, contract=contract,
                memory_limit=memory_limit)
        else:
            h_bp += somf_two_electron(mol, p_eff, contract=contract,
                                      memory_limit=memory_limit)

    if convention == "i":
        h = SOMF_ALPHA2_HALF * h_bp
        if use_ecp:
            h = h - ecp_soc(mol)
    else:
        h = h_bp
        if use_ecp:
            h = h + ecp_soc(mol)

    if check:
        asym = float(np.abs(h + h.transpose(0, 2, 1)).max())
        if asym > 1e-10:
            raise ValueError(
                f"operator is not antisymmetric: max|h + h^T| = {asym:.3e}")
    return h


def somf_variants(mol, P, contract="auto", memory_limit=None):
    """Named ECP SOMF operator variants in one common convention.

    Every returned matrix is in the ``+i`` convention
    ``H_SO = i sum_k h_k s_k``, so a caller can assemble the state
    interaction matrix with the same prefactor for every variant.

    The variants span the decomposition freedom of an ECP:

    ``ecp``
        ``ECPso`` only.  This is the one-electron operator fixed by the
        ECP fit; no explicit Breit-Pauli term is added.
    ``nuclear_non_ecp``
        Breit-Pauli nuclear SOC on centers without a spin-orbit ECP.
    ``breit_pauli``
        ``nuclear_non_ecp`` plus the full valence ``J - 3/2 K`` mean
        field, without ``ECPso``.
    ``valence``
        The full valence SOMF definition of
        :func:`valence_somf_matrix` (the module default).
    ``one_center``
        ``valence`` with the two-electron kernel and the density
        truncated to the spin-orbit ECP centers.
    ``atomic_subtracted``
        ``valence`` with the on-site spin-orbit ECP density removed from
        the mean field.  This is the variant that assumes the ECP
        already contains the free-atom valence mean field, so only the
        molecular (inter-site) change is added.
    ``point_charge``
        Scalar-ECP point-charge model: effective-charge nuclear SOC on
        every center and no ``ECPso``.
    ``double``
        ``ECPso`` plus the nuclear SOC on every center and the full
        valence mean field.  Diagnostic view of the double counting.

    Returns
    -------
    dict of str -> ndarray, shape (3, nao, nao)
    """
    kw = dict(contract=contract, memory_limit=memory_limit, convention="i")
    out = {}
    out["ecp"] = valence_somf_matrix(
        mol, P, nuclear="none", ecp_term=True, two_electron="none",
        one_electron="ecpso", **kw)
    out["nuclear_non_ecp"] = valence_somf_matrix(
        mol, P, nuclear="auto", ecp_term=False, two_electron="none", **kw)
    out["breit_pauli"] = valence_somf_matrix(
        mol, P, nuclear="auto", ecp_term=False, two_electron="full", **kw)
    out["valence"] = valence_somf_matrix(
        mol, P, nuclear="auto", ecp_term=True, two_electron="full", **kw)
    out["one_center"] = valence_somf_matrix(
        mol, P, nuclear="auto", ecp_term=True, two_electron="one_center",
        **kw)
    out["atomic_subtracted"] = valence_somf_matrix(
        mol, P, nuclear="auto", ecp_term=True, two_electron="full",
        onsite="subtract", **kw)
    out["point_charge"] = valence_somf_matrix(
        mol, P, nuclear="all", ecp_term=False, two_electron="full",
        one_electron="point_charge", **kw)
    out["double"] = valence_somf_matrix(
        mol, P, nuclear="all", ecp_term=True, two_electron="full", **kw)
    return out


def somf_density_response(mol, D, contract="explicit"):
    """Density-response matrix of the two-electron SOMF term.

    For ``T = sum_pq D_pq h2[P]_pq`` the derivative with respect to the
    reference density is

        dT = sum_rs M_rs dP_rs,
        M_rs = sum_pq D_pq [ g[p,q,r,s] - 3/2 g[p,r,s,q]
                             - 3/2 g[s,q,p,r] ].

    The kernel is not symmetric under ``(pq) <-> (rs)``, so ``M`` is not
    ``somf_two_electron(mol, D)``; this function performs the correct
    contraction.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    D : array_like, shape (nao, nao)
        Density used in the matrix element.  For an SOC matrix element
        between two real scalar states this is the transition 1-RDM and
        is antisymmetric for the degenerate pair.
    contract : {'explicit'}
        Only the explicit route is implemented.  A direct (integral-
        driven) route needs a dedicated contraction of the SOC kernel
        and is not part of this value layer.

    Returns
    -------
    ndarray, shape (3, nao, nao)
        ``M`` in the same raw convention as ``int2e_p1vxp1``.
    """
    if contract != "explicit":
        raise NotImplementedError(
            "somf_density_response currently supports contract='explicit' "
            "only; a direct contraction of the SOC kernel is not available")
    nao = mol.nao_nr()
    D = np.asarray(D, dtype=float)
    if D.shape != (nao, nao):
        raise ValueError(f"D must have shape ({nao}, {nao}), got {D.shape}")
    g = np.asarray(mol.intor("int2e_p1vxp1", aosym="s1"),
                   dtype=float).reshape(3, nao, nao, nao, nao)
    m = np.einsum("pq,kpqrs->krs", D, g, optimize=True)
    m -= 1.5 * np.einsum("pq,kprsq->krs", D, g, optimize=True)
    m -= 1.5 * np.einsum("pq,ksqpr->krs", D, g, optimize=True)
    return m


def somf_definition(mol):
    """Human- and machine-readable definition of the operator in use.

    The returned dict records which centers contribute which term, the
    electron count of the reference density, and the approximations that
    are not fixed by the ECP itself.
    """
    so = so_ecp_centers(mol)
    nuc = nuclear_centers(mol)
    scalar_ecp = [ia for ia in range(mol.natm)
                  if ia not in so and ia in set(_ecp_symbols(mol))]
    return {
        "so_ecp_centers": so,
        "nuclear_centers": nuc,
        "scalar_ecp_centers": scalar_ecp,
        "density_electrons": int(mol.nelectron),
        "one_electron": ("ECPso on SO-ECP centers; Z_eff nuclear SOC "
                         "on the other centers"),
        "two_electron": "valence J - 3/2 K",
        "convention": "+i (H_SO = i sum_k h_k s_k)",
        "variants": [
            "ecp", "nuclear_non_ecp", "breit_pauli", "valence",
            "one_center", "atomic_subtracted", "point_charge", "double",
        ],
        "approximations": [
            "the ECP spin-orbit part is assumed to contain the core "
            "contribution to the one-electron SOC of its center",
            "the explicit valence mean field may double count the "
            "valence-valence SOC already fitted into the ECP; the "
            "variants returned by somf_variants() span this freedom",
            "atomic_subtracted removes the on-site block of the valence "
            "density from the mean field, which assumes the ECP already "
            "contains the free-atom valence mean field",
        ],
    }


def _ecp_symbols(mol):
    return list(getattr(mol, "_ecp", {}) or {})
