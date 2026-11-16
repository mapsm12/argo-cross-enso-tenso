#!/bin/bash
# =============================================================================
# MASTER REPRODUCIBILITY SCRIPT - ENSO-TENSOR SCIENTIFIC AUDIT
# Author: M. Andrade (Instituto Geofísico del Perú - IGP / UNSAAC)
# Date: November 2026
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_EXEC="/home/elnino2/miniforge3/envs/oceanMiguel/bin/python"
TEX_EXEC="/data/users/cah/ELNINO/elnino2/proyectos/TESIS-POS/REVISION/2026/TOOLS/texlive/2026/bin/x86_64-linux/pdflatex"
BIB_EXEC="/data/users/cah/ELNINO/elnino2/proyectos/TESIS-POS/REVISION/2026/TOOLS/texlive/2026/bin/x86_64-linux/bibtex"

echo "========================================================================="
echo "STARTING FULL REPRODUCIBILITY REPLICATION FOR ENSO-TENSOR MANUSCRIPT"
echo "Working Directory: ${BASE_DIR}"
echo "Python Executable: ${PYTHON_EXEC}"
echo "========================================================================="

echo "[1/7] Running Phase 1: Observational Data and Timeline Audit..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_01_data_and_reproducibility.py"

echo "[2/7] Running Phase 2: Mathematical Consistency and Algebraic Unit Tests..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_02_mathematical_consistency.py"

echo "[3/7] Running Phase 3 & 4: Temporal Irreversibility and Null Models..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_03_irreversibility_and_null_models.py"

echo "[4/7] Running Phase 5: Extremes, Jackknife Subsampling and Composites..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_04_extremes_jackknife_composites.py"

echo "[5/7] Running Phase 6: Out-of-Sample Predictive Evaluation (M1–M5)..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_05_predictive_evaluation.py"

echo "[6/7] Running Phase 7: Bibliography Audit and DOI Verification..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/audit_06_bibliography_and_originality.py"

echo "[7/7] Regenerating All Figures 1–6 (and Figure 7)..."
"${PYTHON_EXEC}" "${BASE_DIR}/scripts/generate_all_final_figures.py"

echo "========================================================================="
echo "COMPILING LATEX MANUSCRIPT: ENSO_Tensor_FINAL.tex -> ENSO_Tensor_FINAL.pdf"
echo "========================================================================="
cd "${BASE_DIR}"
"${TEX_EXEC}" -interaction=nonstopmode ENSO_Tensor_FINAL.tex > /dev/null
"${BIB_EXEC}" ENSO_Tensor_FINAL > /dev/null
"${TEX_EXEC}" -interaction=nonstopmode ENSO_Tensor_FINAL.tex > /dev/null
"${TEX_EXEC}" -interaction=nonstopmode ENSO_Tensor_FINAL.tex > /dev/null

echo "========================================================================="
echo "ALL AUDIT EXPERIMENTS, TABLES, FIGURES, AND PDF MANUSCRIPT ARE REPRODUCED!"
echo "Target PDF: ${BASE_DIR}/ENSO_Tensor_FINAL.pdf"
echo "========================================================================="
