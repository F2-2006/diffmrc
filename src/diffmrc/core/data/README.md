# Bundled data

## `sap_grasp_large.json`

Superposition-of-atomic-potentials (SAP) fit used by
`diffmrc.core.x2c_sap` for two-electron picture-change corrections.

- Source: Basis Set Exchange, `sap_grasp_large` (version 1).
- Original fit: S. Lehtola, L. Visscher, E. Engel, J. Chem. Phys. 2020,
  152, 144105.
- The file stores, for each element, the exponents and coefficients of
  unit-integral s Gaussians whose potential is
  `V_A(r) = -Z_A/r + sum_k c_k erf(sqrt(alpha_k) r)/r`.
- Basis Set Exchange citation: B. P. Pritchard, D. Altarawy, B. Didier,
  T. D. Windus, T. L. Windus, J. Chem. Inf. Model. 2019, 59, 4814.

The data are redistributed here for reproducibility of the SAP-X2C value
layer. Users are responsible for compliance with the original Basis Set
Exchange terms; cite the two references above in derived work.
