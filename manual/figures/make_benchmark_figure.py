"""Deviation figure for the QUEST-subset benchmark (Tutorial, Fig. 1).

Data: the fourteen states of the 7-molecule run, deviations in eV
against the QUEST best estimate and against its sc-NEVPT2 column.
Source of every number: the recorded runs under the project's
server/results/ (t315/t317/t321 JSON, assembled by t316); the same
values are quoted in the tutorial's benchmark table.

Outputs benchmark_bias.pdf (for the LaTeX build) and
benchmark_bias.png (for inspection and the repository).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (system, state, family, d_vs_tbe, d_vs_sc_nevpt2); top to bottom.
STATES = [
    ("Acetaldehyde", "$^1A''$", "npi", 0.277, 0.161),
    ("Acetaldehyde", "$^3A''$", "npi", 0.151, 0.069),
    ("Glyoxal", "$^1A_u$", "npi", -0.310, -0.423),
    ("Glyoxal", "$^3A_u$", "npi", -0.351, -0.451),
    ("Cyclopropenone", "$^1B_1$", "npi", 0.622, 0.816),
    ("Cyclopropenone", "$^3B_1$", "npi", 0.547, 0.929),
    ("Formamide", "$^1A''$", "npi", 0.105, 0.019),
    ("Formamide", "$^3A''$", "npi", 0.075, 0.003),
    ("Cyanogen", "$^1\\Sigma_u^-$", "ppi", 0.615, 0.620),
    ("Cyanogen", "$^3\\Sigma_u^+$", "ppi", 1.200, 1.185),
    ("Propynal", "$^1A''$", "ppi", 1.349, 1.282),
    ("Propynal", "$^3A'$", "ppi", 0.932, 0.700),
    ("Cyanoacetylene", "$^1\\Sigma_u^-$", "ppi", 1.280, 1.238),
    ("Cyanoacetylene", "$^3\\Sigma_u^+$", "ppi", 1.472, 1.433),
]

COLOR = {"npi": "#1f77b4", "ppi": "#d95f02"}
BAND = {"npi": "#e8f0f8", "ppi": "#fbe9dc"}

n = len(STATES)
fig, ax = plt.subplots(figsize=(7.0, 4.6))
ys = list(range(n - 1, -1, -1))            # first entry on top

# family bands
ax.axhspan(ys[0] + 0.5, ys[7] - 0.5, color=BAND["npi"], zorder=0)
ax.axhspan(ys[8] + 0.5, ys[13] - 0.5, color=BAND["ppi"], zorder=0)
ax.axvline(0.0, color="#888888", lw=0.8, ls="--", zorder=1)

for (sysname, state, fam, d_tbe, d_sc), y in zip(STATES, ys):
    ax.plot([d_tbe, d_sc], [y, y], color=COLOR[fam], lw=1.0, alpha=0.55,
            zorder=2)
    ax.scatter(d_tbe, y, s=34, color=COLOR[fam], zorder=3,
               label="vs best estimate" if y == ys[0] else None)
    ax.scatter(d_sc, y, s=34, facecolor="white",
               edgecolor=COLOR[fam], linewidth=1.3, zorder=3,
               label="vs sc-NEVPT2" if y == ys[0] else None)

ax.set_yticks(ys)
ax.set_yticklabels([f"{s}  {st}" for s, st, *_ in STATES], fontsize=8.5)
ax.set_xlabel("deviation from reference (eV)", fontsize=9.5)
ax.set_xlim(-0.65, 1.65)
ax.tick_params(axis="x", labelsize=8.5)

# family labels on the right
ax.text(1.60, ys[3], "$n\\to\\pi^*$", ha="right", va="center",
        fontsize=9.5, color=COLOR["npi"], fontweight="bold")
ax.text(1.60, ys[10], "$\\pi\\to\\pi^*$", ha="right", va="center",
        fontsize=9.5, color=COLOR["ppi"], fontweight="bold")

ax.legend(loc="upper left", fontsize=8.5, frameon=False)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)

fig.tight_layout()
fig.savefig("benchmark_bias.pdf")
fig.savefig("benchmark_bias.png", dpi=200)
print("figure written: benchmark_bias.pdf / .png")
