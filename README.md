# ARGO-CROSS ENSO–Tensor

Reproducible code, synchronized observational tables, audited outputs, and LaTeX manuscript sources for:

> **Multilinear and Subsurface Evaluation of the ENSO $E$ and $C$ Indices: Quantifying Residual Dynamical Information, Temporal Irreversibility, and Threshold Sensitivity in the Equatorial Pacific**
>
> M. Andrade  
> *Dirección de Ciencias de la Atmósfera, Hidrósfera y Cambio Climático, Instituto Geofísico del Perú (IGP), Lima, Perú*  
> Correspondence: `mandrade@igp.gob.pe`

---

## Repository Structure

```text
argo-cross-enso-tensor/
├── data/
│   └── tabla_datos_sincronizados_1980_2018.csv   # Synchronized monthly E, C, D20, and taux (1980-03 to 2018-12, N = 466)
├── scripts/
│   ├── audit_01_data_and_reproducibility.py      # Observational timeline & epoch sensitivity (1980-2018 vs 1985-2018)
│   ├── audit_02_mathematical_consistency.py      # Third-order tensor norms, Frobenius balance & Gram-Schmidt identity
│   ├── audit_03_irreversibility_and_null_models.py # Bi-lagged T_ECH(tau1, tau2), Fourier/VAR(2)/quadratic null tests
│   ├── audit_04_extremes_jackknife_composites.py # Super El Niño Jackknife sensitivity & event composites
│   ├── audit_05_predictive_evaluation.py         # Out-of-sample M1-M5 RMSE (2006-2018) & Diebold-Mariano tests
│   ├── audit_06_bibliography_and_originality.py  # Reference & positioning checks
│   ├── audit_07_master_numerical_reconciliation.py # Cross-check of all reported tensor components and norms
│   └── generate_all_final_figures.py             # Generates Figures 1-7 in PNG and PDF
├── tablas/                                       # Audited summary CSV tables
├── audits/                                       # Reproducible JSON/TXT audit outputs
├── figuras/                                      # Final publication figures (fig01 - fig07, PDF & PNG)
├── manuscript/
│   ├── ENSO_Tensor_FINAL.tex                     # Master LaTeX manuscript
│   ├── references_final.bib                      # BibTeX bibliography
│   └── ENSO_Tensor_FINAL.pdf                     # Compiled manuscript PDF
└── requirements.txt                              # Python dependencies
```

---

## Data Provenance

- **Surface ENSO Indices ($E, C$):** Derived from NOAA ERSSTv5 following Takahashi et al. (2011).
- **Subsurface Thermocline Depth ($D20$):** NOAA PMEL TAO/TRITON moored array (`iso20_complete.nc`, EPOCS/TOGA/ATLAS eras) integrated and cross-validated with ECMWF ORAS5 ocean reanalysis.
- **Zonal Wind Stress ($\tau_x$):** ECMWF ORAS5 (`sozotaux`, eastward wind stress on the ocean surface in $\mathrm{N/m^2}$) averaged over $2^\circ\text{S}$–$2^\circ\text{N}$, $160^\circ\text{E}$–$160^\circ\text{W}$ (validated against ASCAT/Blended winds, $r = +0.9757$, and ERA5 `avg_iews`, $r = +0.9856$).

---

## Reproducing the Analyses

Install Python 3.11+ dependencies:

```bash
pip install -r requirements.txt
```

> **Note:** Scripts `audit_01`–`audit_07` and `generate_all_final_figures.py` reference the local synchronized table (`data/tabla_datos_sincronizados_1980_2018.csv`) and, for raw Hovmöller panels in Figure 4, the full TAO/ORAS5 NetCDF grids. All summary statistics, tensor decompositions, surrogate null tests ($B = 2000$, seed `42`), Jackknife experiments, and predictive evaluations run directly from the synchronized CSV table included in `data/`.

---

## License

MIT License (see `LICENSE`).
