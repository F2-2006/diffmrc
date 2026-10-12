"""Bidirectional exact-priced growth of an active-space window.

The greedy growth loop behind the f-block experiments: from a seed
window, each step prices two candidate moves and takes the better one.

    up:   ncore,     nact + 1   -- extend the window contiguously
                                   upward, taking the next column in the
                                   energy order;
    down: ncore - 1, nact + 1   -- pull the top core orbital into the
                                   window. This is the move that can
                                   reach a frozen f shell: the t84
                                   composition scan showed the 4f block
                                   sits inside the frozen core for most
                                   f-block windows, and an up-only rule
                                   structurally cannot discover it.

Pricing is delegated: the caller passes an energy oracle mapping
(ncore, nact, column order) to a total energy -- exact CASCI in the
production runner (``code/t85_fblock_growth.py``), synthetic functions
in the tests. This module never touches an electronic-structure object,
so it lives in the policy layer; the energy unit is whatever the oracle
returns.

Stopping.  The historical rule (the recorded t85 runs) stops when a
step's best gain falls below ``f_stop`` times the *first* step's gain.
That rule can be fooled by complementarity: two orbitals worth much
more jointly than apart produce a small marginal step after a large
first step (the N2 pi-pair; ``experiments/greedy_growth_v2.py``).
``stop_rule="step"`` compares against the previous step's gain instead,
which is immune to that pattern. Both are provided, and the default
keeps the historical behavior so recorded runs reproduce exactly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Sequence

#: Default cap on nact and the historical stopping fraction.
NACT_CAP_DEFAULT = 10
F_STOP_DEFAULT = 0.05

#: Oracle signature: (ncore, nact, column order) -> total energy.
EnergyOracle = Callable[[int, int, Sequence[int]], float]


@dataclass
class GrowthStep:
    """One recorded move. ``gain`` units are the oracle's energy units.

    The direction, the entering column (by ORIGINAL column index) and
    the window state before the move are recorded; ``gain_down`` is
    ``-inf`` when the down move is not available (ncore <= 1).
    """

    direction: str          # "up" or "down"
    column: int             # original index of the entering column
    gain: float             # gain of the chosen move
    gain_up: float
    gain_down: float
    ncore: int              # state BEFORE the move
    nact: int


@dataclass
class GrowthResult:
    ncore0: int
    nact0: int
    ncore_final: int
    nact_final: int
    steps: List[GrowthStep] = field(default_factory=list)


def grow_window(evaluate: EnergyOracle,
                ncore0: int,
                nact0: int,
                n_columns: int,
                cap: int = NACT_CAP_DEFAULT,
                f_stop: float = F_STOP_DEFAULT,
                stop_rule: str = "first") -> GrowthResult:
    """Run the bidirectional growth loop.

    ``evaluate(ncore, nact, order)`` is called once per candidate window;
    ``order`` is the full column ordering (a permutation of range
    ``n_columns``) whose window block is ``order[ncore:ncore + nact]``.
    The returned steps carry the pre-move state, exactly as the original
    experiment recorded them.
    """
    if stop_rule not in ("first", "step"):
        raise ValueError(f"stop_rule must be 'first' or 'step', "
                         f"got {stop_rule!r}")
    ncore, nact = int(ncore0), int(nact0)
    order = list(range(n_columns))
    steps: List[GrowthStep] = []
    marg_first = None
    while nact < cap:
        e_cur = evaluate(ncore, nact, order)
        vir_lo = ncore + nact
        # up: contiguous extension
        if vir_lo < n_columns:
            gain_up = e_cur - evaluate(ncore, nact + 1, order)
        else:
            gain_up = float("-inf")
        # down: pull column (ncore - 1) up across the window
        if ncore > 1:
            j = ncore - 1
            perm = (list(range(j)) + list(range(j + 1, vir_lo))
                    + [j] + list(range(vir_lo, n_columns)))
            order_dn = [order[i] for i in perm]
            gain_dn = e_cur - evaluate(ncore - 1, nact + 1, order_dn)
        else:
            gain_dn, perm = float("-inf"), None
        gain = max(gain_up, gain_dn)
        direction = "up" if gain_up >= gain_dn else "down"
        column = int(order[vir_lo]) if direction == "up" else int(order[ncore - 1])
        if marg_first is None:
            marg_first = gain
        steps.append(GrowthStep(direction=direction, column=column,
                                gain=float(gain), gain_up=float(gain_up),
                                gain_down=float(gain_dn),
                                ncore=ncore, nact=nact))
        if stop_rule == "first":
            stop = gain < f_stop * marg_first
        else:
            prev = steps[-2].gain if len(steps) > 1 else None
            stop = (prev is not None and prev > 0.0
                    and 0.0 < gain < f_stop * prev)
        if stop:
            break
        if direction == "up":
            nact += 1
        else:
            order = order_dn
            ncore -= 1
            nact += 1
    return GrowthResult(ncore0=int(ncore0), nact0=int(nact0),
                        ncore_final=ncore, nact_final=nact, steps=steps)
