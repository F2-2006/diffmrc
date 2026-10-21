# Examples

Small, self-contained scripts for the public interfaces. All three run
from the repository root with the package importable:

```bash
PYTHONPATH=src python examples/ri_df.py
PYTHONPATH=src python examples/ri_nevpt2.py
PYTHONPATH=src python examples/performance_options.py
```

| Script | What it demonstrates |
|---|---|
| `ri_df.py` | RI factors, assembly against the pyscf DF reconstruction and the dense ERI, RI-CASCI, and the three-center and ERI derivatives against central differences. |
| `ri_nevpt2.py` | RI-NEVPT2 correlation energy against the dense four-center path on the same CASCI reference. |
| `performance_options.py` | `ci_solver='auto'` routing, the four-step MO transform against the one-shot contraction, and the array-argument jit form. |

The scripts use H2O/6-31G (CAS(2,2), `ncore=4`) and print the measured
quantities of the run; the expected orders of magnitude are recorded in
the Tutorial, Chapters 10 and 11.
