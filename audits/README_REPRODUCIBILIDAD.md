# PROTOCOLO DE REPRODUCIBILIDAD CIENTÍFICA: MANUSCRITO ENSO–TENSOR

**Autor:** M. Andrade  
**Filiación:** Instituto Geofísico del Perú (IGP) / Universidad Nacional de San Antonio Abad del Cusco (UNSAAC)  
**Fecha:** Noviembre 2026  
**Directorio Base:** `/home/elnino2/datos/proyectos/ARGO-CROSS/REVISION_FINAL_ENSO_TENSOR`

---

## 1. Estructura de Directorios

```
REVISION_FINAL_ENSO_TENSOR/
├── ENSO_Tensor_FINAL.tex               # Código fuente LaTeX del artículo final (21 págs)
├── ENSO_Tensor_FINAL.pdf               # PDF compilado de calidad editorial con figuras y bibliografía
├── references_final.bib                # 22 referencias verificadas vía Crossref API
├── 01_AUDITORIA_DATOS_Y_REPRODUCIBILIDAD.md
├── 02_AUDITORIA_MATEMATICA.md
├── 03_VALIDACION_MODELOS_NULOS.md
├── 04_EVENTOS_Y_ROBUSTEZ.md
├── 05_BIBLIOGRAFIA_Y_ORIGINALIDAD.md
├── 06_VALIDACION_PREDICTIVA.md
├── RESPUESTA_A_OBSERVACIONES.md        # Matriz punto a punto respondiendo a las críticas
├── VEREDICTO_CIENTIFICO_FINAL.md       # Veredicto simulado de 3 revisores independientes
├── LIMITACIONES_PENDIENTES.md          # Límites del estudio y agenda futura
├── FIGURAS_FINALES/                    # Figuras vectoriales (PDF) y de alta resolución (PNG)
│   ├── fig01_joint_distributions_ECD20.pdf/.png
│   ├── fig02_linear_vs_nonlinear_decomposition.pdf/.png
│   ├── fig03_lagged_tensor_surface_ECH.pdf/.png
│   ├── fig04_hovmoller_wind_thermocline_coupling.pdf/.png
│   ├── fig05_predictive_evaluation_out_of_sample.pdf/.png
│   ├── fig06_jackknife_extreme_event_collapse.pdf/.png
│   └── fig07_composiciones_eventos_ENSO.pdf/.png
├── tablas/                             # Tablas numéricas auditadas en CSV
│   ├── tabla_inventario_reproducibilidad.csv
│   ├── tabla_componentes_tensor_auditada.csv
│   ├── tabla_modelos_nulos_auditada.csv
│   ├── tabla_autocorrelacion_C_auditada.csv
│   ├── tabla_jackknife_auditada.csv
│   └── tabla_evaluacion_predictiva_auditada.csv
├── scripts/                            # Scripts modulares de Python
│   ├── audit_01_data_and_reproducibility.py
│   ├── audit_02_mathematical_consistency.py
│   ├── audit_03_irreversibility_and_null_models.py
│   ├── audit_04_extremes_jackknife_composites.py
│   ├── audit_05_predictive_evaluation.py
│   ├── audit_06_bibliography_and_originality.py
│   └── generate_all_final_figures.py
└── RESULTADOS_REPRODUCIBLES/           # Paquete de replicación automatizada
    ├── run_all_audits.sh
    └── README_REPRODUCIBILIDAD.md
```

---

## 2. Instrucciones de Replicación de un Solo Paso

Para reproducir todos los experimentos, recalcular las tablas CSV, generar todas las figuras de publicación y recompilar el PDF completo, ejecute:

```bash
cd /home/elnino2/datos/proyectos/ARGO-CROSS/REVISION_FINAL_ENSO_TENSOR/RESULTADOS_REPRODUCIBLES
./run_all_audits.sh
```

---

## 3. Entorno de Ejecución y Dependencias

- **Intérprete Python:** `/home/elnino2/miniforge3/envs/oceanMiguel/bin/python` (Python 3.10+)
- **Paquetes clave:** `numpy`, `scipy`, `pandas`, `xarray`, `netCDF4`, `matplotlib`, `statsmodels`.
- **Motor LaTeX:** TeX Live 2026 (`pdflatex`, `bibtex`).

---

## 4. Semillas Pseudoaleatorias Deterministas

Para asegurar la reproducibilidad bit a bit de todos los modelos estocásticos y contrastes de hipótesis:
- **Bootstrap en Bloques (L=18m, B=2000):** `seed = 42`
- **Series Sustitutas Multivariadas de Fourier (B=2000):** `seed = 100`
- **Modelo Nulo Trivariado VAR(2) (B=2000):** `seed = 2026`
- **Modelo Nulo Dinámico de Recarga Acoplada no Lineal (B=1000):** `seed = 777`
- **Búsqueda Exploratoria de Retardos (625 combinaciones):** `seed = 999`
