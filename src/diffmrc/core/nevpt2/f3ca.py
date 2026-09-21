"""Fourth-order density contraction for the f3ca and f3ac intermediates.

The contraction was derived from the semantics of the reference C
implementation and validated against it to machine precision on H2 CAS(2,2).

    t1[s, i, a]       = <det_a[stra_id], det_b[strb_id+s] | E^i_a | ci>
    t2[k, i, a, j, l] = <det_a[stra_id], det_b[strb_id+k] | E^i_a E^j_l | ci>

    gt2_f3ca[k, i, p] = sum_{a,j,l} eri[p,a,j,l] t2[k,i,a,j,l]
    gt2_f3ac[k, a, p] = sum_{q,r,s} eri[p,q,r,s] t2[k,a,q,r,s]

    rdm3[j,i,k,l,p,q] = sum_n gt2[n,(p,q)] t2ket[n,l,k,i,j],  k <= j
    rdm2_raw[col=(i,a), row=(p,q)] = sum_n gt2[n,(p,q)] t1ket[n,(i,a)]

    fdm2[(i,j),.] = raw[(j,i),.]          block transposition at the end of the
                                          reference contraction
    fdm3          : particle symmetrization over the trailing loop

The numpy version defined here is the reference for the differentiable version
below it.
"""
import numpy as np


_AB_MAP_CACHE = {}


def ab_index_map(ncore=0, nact=2, nele_act=2):
    """Static index map: ci_ab[i, j] = ci_flat[idx_ab[i, j]].

    The gather is differentiable and works under JAX. The map is cached at
    module level and reused while ncore, nact, and nele_act are unchanged.
    """
    key = (ncore, nact, nele_act)
    if key in _AB_MAP_CACHE:
        return _AB_MAP_CACHE[key]
    from itertools import combinations
    nso = 2 * (ncore + nact)
    act_so = list(range(2 * ncore, nso))
    neleca = nele_act // 2
    nelecb = nele_act - neleca
    ab_of_det = []
    for comb in combinations(act_so, nele_act):
        det = 0
        for so in list(range(2 * ncore)) + list(comb):
            det |= (1 << so)
        alpha = beta = 0
        for so in act_so:  # active spin orbitals only; the core is doubly occupied
            if (det >> so) & 1:
                if so % 2 == 0:
                    alpha |= (1 << (so // 2 - ncore))
                else:
                    beta |= (1 << ((so - 1) // 2 - ncore))
        ab_of_det.append((alpha, beta))
    proj = [(a, b) for (a, b) in ab_of_det
            if bin(a).count('1') == neleca and bin(b).count('1') == nelecb]
    # The (i, j) index semantics of this map follow the cistring layout of the
    # reference implementation, which orders strings by numeric value:
    #
    #     make_strings((0,1,2,3), 2) -> [0b11, 0b101, 0b110, 0b1001, ...]
    #
    # This is the same as sorted(), but for nact >= 4 it differs from the
    # lexicographic order of itertools.combinations; with nact = 4 the first
    # disagreement appears at index 2, where sorted() gives 0b110 and
    # combinations gives 0b1001.
    #
    # An earlier version of this function used sorted() directly and produced
    # correct results. Calling the reference cistring routine instead makes the
    # assumption of agreement visible in the code, and the set assertions below
    # guard against regressions.
    #
    # Note that the CI solver in this package enumerates determinants in
    # combinations order, so this function performs the conversion between the
    # two conventions.
    from pyscf.fci import cistring as _cistr
    alpha_strings = [int(x) for x in
                     _cistr.gen_strings4orblist(range(nact), neleca)]
    beta_strings = [int(x) for x in
                    _cistr.gen_strings4orblist(range(nact), nelecb)]
    assert set(alpha_strings) == {a for a, _ in proj}, 'alpha string set mismatch'
    assert set(beta_strings) == {b for _, b in proj}, 'beta string set mismatch'
    ia_of = {s: i for i, s in enumerate(alpha_strings)}
    ib_of = {s: i for i, s in enumerate(beta_strings)}
    idx_ab = np.full((len(alpha_strings), len(beta_strings)), -1)
    for det_idx, (a, b) in enumerate(ab_of_det):
        if bin(a).count('1') == neleca and bin(b).count('1') == nelecb:
            idx_ab[ia_of[a], ib_of[b]] = det_idx
    assert (idx_ab >= 0).all(), 'alpha-beta determinant map is incomplete'
    # Fermionic phase: the interleaved spin-orbital bitmap used by the CI solver
    # versus the alpha-beta product basis |alpha x beta>. Each beta_j must be
    # moved past every occupied alpha_k with k > j, and each transposition
    # contributes a factor of -1.
    phase_ab = np.ones((len(alpha_strings), len(beta_strings)))
    for (i, a) in enumerate(alpha_strings):
        for (j, b) in enumerate(beta_strings):
            ph = 1
            for jj in range(nact):
                if (b >> jj) & 1:
                    n_alpha_after = bin(a & ~((1 << (jj + 1)) - 1)).count('1')
                    if n_alpha_after % 2:
                        ph = -ph
            phase_ab[i, j] = ph
    _AB_MAP_CACHE[key] = (idx_ab, phase_ab)
    return idx_ab, phase_ab


def ci_flat_to_ab(ci, ncore=0, nact=2, nele_act=2):
    """Convert the CI vector from the CI-solver layout to the alpha-beta layout.

    The input is the full-determinant one-dimensional CI vector, with
    determinants enumerated in combinations order over interleaved spin orbitals
    (alpha at 2k, beta at 2k+1). The output is the (na, nb) layout of the
    reference implementation, an outer product of alpha and beta strings in
    numeric bitmap order.

    The CI vector carries a global phase, and since the intermediate t2ket is
    linear in it, the element-wise signs must agree with the reference, which
    requires the combination order and the bitmap order to match on both sides.
    """
    from itertools import combinations
    nso = 2 * (ncore + nact)
    act_so = list(range(2 * ncore, nso))
    dets = []
    ab_of_det = []
    for comb in combinations(act_so, nele_act):
        det = 0
        for so in list(range(2 * ncore)) + list(comb):
            det |= (1 << so)
        dets.append(det)
        # The alpha bitmap covers even active spin orbitals, the beta bitmap
        # the odd ones; the core does not enter either.
        alpha = 0
        beta = 0
        for so in act_so:
            if (det >> so) & 1:
                if so % 2 == 0:
                    alpha |= (1 << (so // 2 - ncore))
                else:
                    beta |= (1 << ((so - 1) // 2 - ncore))
        ab_of_det.append((alpha, beta))
    neleca = nele_act // 2
    nelecb = nele_act - neleca
    proj = [(a, b) for (a, b) in ab_of_det
            if bin(a).count('1') == neleca and bin(b).count('1') == nelecb]
    # The (i, j) index semantics of this map follow the cistring layout of the
    # reference implementation, which orders strings by numeric value:
    #
    #     make_strings((0,1,2,3), 2) -> [0b11, 0b101, 0b110, 0b1001, ...]
    #
    # This is the same as sorted(), but for nact >= 4 it differs from the
    # lexicographic order of itertools.combinations; with nact = 4 the first
    # disagreement appears at index 2, where sorted() gives 0b110 and
    # combinations gives 0b1001.
    #
    # An earlier version of this function used sorted() directly and produced
    # correct results. Calling the reference cistring routine instead makes the
    # assumption of agreement visible in the code, and the set assertions below
    # guard against regressions.
    #
    # Note that the CI solver in this package enumerates determinants in
    # combinations order, so this function performs the conversion between the
    # two conventions.
    from pyscf.fci import cistring as _cistr
    alpha_strings = [int(x) for x in
                     _cistr.gen_strings4orblist(range(nact), neleca)]
    beta_strings = [int(x) for x in
                    _cistr.gen_strings4orblist(range(nact), nelecb)]
    assert set(alpha_strings) == {a for a, _ in proj}, 'alpha string set mismatch'
    assert set(beta_strings) == {b for _, b in proj}, 'beta string set mismatch'
    ia_of = {s: i for i, s in enumerate(alpha_strings)}
    ib_of = {s: i for i, s in enumerate(beta_strings)}
    na, nb = len(alpha_strings), len(beta_strings)
    idx_ab, phase_ab = ab_index_map(ncore, nact, nele_act)
    return (ci[idx_ab] * phase_ab)


def t1_py(ci, stra_id, strb_id, bcount, norb, links_a, links_b, na, nb):
    """t1[s, i, a] = <det_a[stra_id], det_b[strb_id+s] | E^i_a | ci>.

    Accumulated through the link lists, reproducing the two reference kernels
    for the beta and alpha parts respectively.
    """
    t1 = np.zeros((bcount, norb * norb))
    for str0 in range(bcount):
        for (a, i, addr, sign) in links_b[strb_id + str0]:
            t1[str0, i * norb + a] += sign * ci[stra_id, addr]
    for (a, i, addr, sign) in links_a[stra_id]:
        for k in range(bcount):
            t1[k, i * norb + a] += sign * ci[addr, strb_id + k]
    return t1


def t1_jax(ci, stra_id, strb_id, bcount, norb, links_a, links_b, na, nb):
    """Vectorized t1 using functional scatter, differentiable under JAX."""
    import jax.numpy as jnp
    t1 = jnp.zeros((bcount, norb * norb))
    # beta part: entries (a, i, addr, sign) of links_b[s], with s = strb_id + str0
    for j in range(len(links_b[0])):
        a = jnp.asarray([links_b[strb_id + s][j][0] for s in range(bcount)])
        i = jnp.asarray([links_b[strb_id + s][j][1] for s in range(bcount)])
        addr = jnp.asarray([links_b[strb_id + s][j][2] for s in range(bcount)])
        sign = jnp.asarray([links_b[strb_id + s][j][3] for s in range(bcount)])
        t1 = t1.at[(jnp.arange(bcount), i * norb + a)].add(sign * ci[stra_id, addr])
    # alpha part: entries (a, i, addr, sign) of links_a[stra_id]
    for (a, i, addr, sign) in links_a[stra_id]:
        t1 = t1.at[:, i * norb + a].add(sign * ci[addr, strb_id:strb_id + bcount])
    return t1


def t2_py(ci, stra_id, strb_id, bcount, norb, links_a, links_b, na, nb):
    """t2[k, i, a, j, l], reproducing the two reference kernels that build it."""
    t2 = np.zeros((bcount, norb, norb, norb, norb))
    t1_all = t1_py(ci, stra_id, 0, nb, norb, links_a, links_b, na, nb)
    for k in range(bcount):
        for (a, i, addr, sign) in links_b[strb_id + k]:
            t2[k, i, a, :, :] += sign * t1_all[addr].reshape(norb, norb)
    for (a, i, addr, sign) in links_a[stra_id]:
        t1p = t1_py(ci, addr, strb_id, bcount, norb, links_a, links_b, na, nb)
        for k in range(bcount):
            t2[k, i, a, :, :] += sign * t1p[k].reshape(norb, norb)
    return t2


def t2_jax(ci, stra_id, strb_id, bcount, norb, links_a, links_b, na, nb):
    """Vectorized t2, differentiable under JAX."""
    import jax.numpy as jnp
    t2 = jnp.zeros((bcount, norb, norb, norb, norb))
    # rdm4_0b_t2
    t1_all = t1_jax(ci, stra_id, 0, nb, norb, links_a, links_b, na, nb).reshape(nb, norb, norb)
    for j in range(len(links_b[0])):
        a = jnp.asarray([links_b[strb_id + k][j][0] for k in range(bcount)])
        i = jnp.asarray([links_b[strb_id + k][j][1] for k in range(bcount)])
        addr = jnp.asarray([links_b[strb_id + k][j][2] for k in range(bcount)])
        sign = jnp.asarray([links_b[strb_id + k][j][3] for k in range(bcount)])
        t2 = t2.at[(jnp.arange(bcount), i, a)].add(sign[:, None, None] * t1_all[addr])
    # rdm4_a_t2
    for (a, i, addr, sign) in links_a[stra_id]:
        t1p = t1_jax(ci, addr, strb_id, bcount, norb, links_a, links_b, na, nb).reshape(bcount, norb, norb)
        t2 = t2.at[:, i, a].add(sign * t1p)
    return t2


def gt2_f3ca(eri, t2ket, norb):
    """gt2[chunk, (i,p)] = Σ_{a,j,l} eri[p,a,j,l]·t2ket[chunk, i, a, j, l]"""
    return np.einsum('pajl,niajl->nip', eri, t2ket).reshape(t2ket.shape[0], norb * norb)


def gt2_f3ac(eri, t2ket, norb):
    """gt2[chunk, (p,a)] = sum_{q,r,s} eri[p,q,r,s] t2ket[chunk, q, a, r, s].

    Reproduces the reference kernel, which transposes the t2 block and calls
    dgemm with the T and N flags. The reference writes in C column-major order,
    so the row-major pair index here is p * norb + a.
    """
    return np.einsum('pqrs,kqars->kpa', eri, t2ket).reshape(t2ket.shape[0], norb * norb)


def assemble_rdm23(gt2, t2ket, norb):
    """Build rdm3 and the raw rdm2 intermediate.

        rdm3[j,i,k,l,p,q] = sum_n gt2[n,(p,q)] t2ket[n,l,k,i,j],   k <= j
        rdm2_raw[col=(i,a), row=(p,q)] = sum_n gt2[n,(p,q)] t1ket[n,(i,a)]
    """
    bcount = gt2.shape[0]
    nnorb = norb * norb
    # Vectorized rdm3, with the k <= j mask that the reference enforces
    # through its loop structure
    gt2_3 = gt2.reshape(bcount, norb, norb)          # [n, p, q]
    t2_5 = t2ket                                  # [n, i, a, j, l]
    rdm3 = np.einsum('npq,nlkij->jiklpq', gt2_3, t2_5)  # [j, i, k, l, p, q]
    mask = (np.arange(norb)[:, None, None, None, None, None]
            >= np.arange(norb)[None, None, :, None, None, None])
    rdm3 = np.where(mask, rdm3, 0.0)
    return rdm3


def assemble_rdm2(gt2, t1ket, norb):
    """raw_rdm2[col=(i,a), row=(p,q)] = Σ_n gt2[n,(p,q)]·t1ket[n,(i,a)]"""
    return np.einsum('ni,np->ip', t1ket, gt2).reshape(norb, norb, norb, norb)


def fdm2fdm3_jax(eri, ci, stra_id_list, strb_id, bcount, norb, links_a, links_b, na, nb,
                 kern='f3ca'):
    """Differentiable fdm2 and fdm3, using functional scatter and einsum throughout."""
    import jax.numpy as jnp
    t2ket = jnp.stack([t2_jax(ci, s, strb_id, bcount, norb, links_a, links_b, na, nb)
                       for s in stra_id_list]).reshape(len(stra_id_list) * bcount,
                                                       norb, norb, norb, norb)
    t1ket = jnp.stack([t1_jax(ci, s, strb_id, bcount, norb, links_a, links_b, na, nb)
                       for s in stra_id_list]).reshape(len(stra_id_list) * bcount, norb * norb)
    if kern == 'f3ca':
        gt2 = jnp.einsum('pajl,niajl->nip', eri, t2ket).reshape(t2ket.shape[0], norb * norb)
    else:
        gt2 = jnp.einsum('pqrs,kqars->kpa', eri, t2ket).reshape(t2ket.shape[0], norb * norb)
    # rdm3[j,i,k,l,p,q] = Σ_n gt2[n,(p,q)]·t2ket[n,l,k,i,j]，k ≤ j
    gt2_3 = gt2.reshape(t2ket.shape[0], norb, norb)
    rdm3 = jnp.einsum('npq,nlkij->jiklpq', gt2_3, t2ket)
    mask = (jnp.arange(norb)[:, None, None, None, None, None]
            >= jnp.arange(norb)[None, None, :, None, None, None])
    rdm3 = jnp.where(mask, rdm3, 0.0)
    # raw rdm2[col=(i,a), row=(p,q)]
    rdm2_raw = jnp.einsum('ni,np->ip', t1ket, gt2).reshape(norb, norb, norb, norb)
    fdm2 = rdm2_raw.transpose(1, 0, 2, 3)
    # Particle symmetrization, applied functionally
    fdm3 = rdm3
    for i in range(norb):
        for j in range(i):
            fdm3 = fdm3.at[j, :, i].set(fdm3[i, :, j].transpose(1, 0, 2, 3))
            fdm3 = fdm3.at[j, i, i].add(fdm2[j])
            fdm3 = fdm3.at[j, :, i, j].add(-fdm2[i])
    return fdm2, fdm3


def fdm2fdm3(eri, ci, stra_id_list, strb_id, bcount, norb, links_a, links_b, na, nb,
             kern='f3ca', t1py=t1_py, t2py=t2_py, npmod=np):
    """Full f3ca/f3ac assembly including particle symmetrization.

    Returns (fdm2, fdm3) in the layout produced by the reference contraction.
    """
    t2ket = npmod.stack([t2py(ci, s, strb_id, bcount, norb, links_a, links_b, na, nb)
                         for s in stra_id_list]).reshape(len(stra_id_list) * bcount, norb, norb, norb, norb)
    t1ket = npmod.stack([t1py(ci, s, strb_id, bcount, norb, links_a, links_b, na, nb)
                         for s in stra_id_list]).reshape(len(stra_id_list) * bcount, norb * norb)
    gt2 = gt2_f3ca(eri, t2ket, norb) if kern == 'f3ca' else gt2_f3ac(eri, t2ket, norb)
    rdm3 = assemble_rdm23(gt2, t2ket, norb)
    rdm2_raw = assemble_rdm2(gt2, t1ket, norb)
    fdm2 = rdm2_raw.transpose(1, 0, 2, 3)  # fdm2[(i,j),·] = raw[(j,i),·]
    fdm3 = rdm3.copy()
    for i in range(norb):
        for j in range(i):
            fdm3[j, :, i] = fdm3[i, :, j].transpose(1, 0, 2, 3)
            fdm3[j, i, i, :] += fdm2[j, :]
            fdm3[j, :, i, j] -= fdm2[i, :]
    return fdm2, fdm3
