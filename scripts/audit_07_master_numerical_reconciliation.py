#!/usr/bin/env python3
"""
MASTER NUMERICAL RECONCILIATION - ENSO TENSOR
Strict internal audit verifying:
1. Exactly 10 components with exact multinomial multiplicities matching full 3x3x3 tensor.
2. Projection identities: T_ECH^lin = a1*T_EEC + a2*T_ECC, etc.
3. Both ddof=1 and ddof=0 exact values.
4. Validation 2000–2018 moving block bootstrap (95% CI) and p-value.
5. Exact Frobenius norm and balance verified across both vector array and tabular representation.
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR

DATA_PATH = '/home/elnino2/datos/proyectos/ARGO-CROSS/analisis_tensorial_EC/fase_3_dinamica_oceanica/tablas/tabla_datos_sincronizados_1980_2018.csv'

def main():
    print("="*80)
    print("AUDITORÍA NUMÉRICA MAESTRA: RECONCILIACIÓN TOTAL DE TABLA 1 Y NORMAS")
    print("="*80)
    
    df = pd.read_csv(DATA_PATH)
    e_raw = df['Indice_E'].values
    c_raw = df['Indice_C'].values
    h_raw = df['D20_East'].interpolate().values
    N = len(e_raw)
    print(f"Total observaciones: N = {N}")
    
    for ddof_val in [1, 0]:
        print("\n" + "="*80)
        print(f"EVALUACIÓN CON DDOF = {ddof_val}")
        print("="*80)
        
        e = (e_raw - np.mean(e_raw)) / np.std(e_raw, ddof=ddof_val)
        c = (c_raw - np.mean(c_raw)) / np.std(c_raw, ddof=ddof_val)
        h = (h_raw - np.mean(h_raw)) / np.std(h_raw, ddof=ddof_val)
        
        # OLS regression h onto e, c (no intercept because standardized, but check with const)
        X_reg = sm.add_constant(np.column_stack([e, c]))
        mod = sm.OLS(h, X_reg).fit()
        a0, a1, a2 = mod.params[0], mod.params[1], mod.params[2]
        r2 = mod.rsquared
        print(f"Regresión h = {a0:+.4f} + {a1:.4f}*e + {a2:.4f}*c (R^2 = {r2*100:.2f}%)")
        
        # Full 3D tensor
        X = np.column_stack([e, c, h])
        T3 = np.einsum('ti,tj,tk->ijk', X, X, X) / N
        
        # Linear reconstruction tensor
        h_lin = a1 * e + a2 * c
        X_lin = np.column_stack([e, c, h_lin])
        T3_lin = np.einsum('ti,tj,tk->ijk', X_lin, X_lin, X_lin) / N
        T3_res = T3 - T3_lin
        
        # Norms via 3D tensor array
        norm_full = np.sqrt(np.sum(T3**2))
        norm_lin = np.sqrt(np.sum(T3_lin**2))
        norm_res = np.sqrt(np.sum(T3_res**2))
        cross_term = 2.0 * np.sum(T3_lin * T3_res)
        
        print("\n[MÉTODO 1: Array 3x3x3 Tridimensional]")
        print(f"||T||_F:      {norm_full:.5f} (cuadrado: {norm_full**2:.5f})")
        print(f"||T_lin||_F:  {norm_lin:.5f} (cuadrado: {norm_lin**2:.5f}, {norm_lin**2/norm_full**2*100:.2f}%)")
        print(f"2<T_l, T_r>:  {cross_term:.5f} ({cross_term/norm_full**2*100:.2f}%)")
        print(f"||T_res||_F:  {norm_res:.5f} (cuadrado: {norm_res**2:.5f}, {norm_res**2/norm_full**2*100:.2f}%)")
        print(f"Balance: {norm_lin**2 + cross_term + norm_res**2:.5f} == {norm_full**2:.5f} (Diff: {abs(norm_lin**2 + cross_term + norm_res**2 - norm_full**2):.2e})")
        
        # The 10 independent components definition
        # Indices: 0: E, 1: C, 2: H
        comps = [
            ('T_EEE', 0, 0, 0, 1, 'Asimetría E'),
            ('T_CCC', 1, 1, 1, 1, 'Asimetría C'),
            ('T_HHH', 2, 2, 2, 1, 'Asimetría H (D20)'),
            ('T_EEC', 0, 0, 1, 3, 'Momento E²·C'),
            ('T_ECC', 0, 1, 1, 3, 'Momento E·C²'),
            ('T_EEH', 0, 0, 2, 3, 'Momento E²·H'),
            ('T_CCH', 1, 1, 2, 3, 'Momento C²·H'),
            ('T_EHH', 0, 2, 2, 3, 'Momento E·H²'),
            ('T_CHH', 1, 2, 2, 3, 'Momento C·H²'),
            ('T_ECH', 0, 1, 2, 6, 'Interacción E·C·H')
        ]
        
        # Method 2: Norms via 10 components with multiplicity
        sq_full_m2 = sum(w * (T3[i,j,k]**2) for _, i, j, k, w, _ in comps)
        sq_lin_m2  = sum(w * (T3_lin[i,j,k]**2) for _, i, j, k, w, _ in comps)
        sq_res_m2  = sum(w * (T3_res[i,j,k]**2) for _, i, j, k, w, _ in comps)
        cross_m2   = 2.0 * sum(w * T3_lin[i,j,k] * T3_res[i,j,k] for _, i, j, k, w, _ in comps)
        
        print("\n[MÉTODO 2: Suma con Multiplicidades de los 10 Componentes]")
        print(f"||T||_F:      {np.sqrt(sq_full_m2):.5f} (cuadrado: {sq_full_m2:.5f})")
        print(f"||T_lin||_F:  {np.sqrt(sq_lin_m2):.5f} (cuadrado: {sq_lin_m2:.5f}, {sq_lin_m2/sq_full_m2*100:.2f}%)")
        print(f"2<T_l, T_r>:  {cross_m2:.5f} ({cross_m2/sq_full_m2*100:.2f}%)")
        print(f"||T_res||_F:  {np.sqrt(sq_res_m2):.5f} (cuadrado: {sq_res_m2:.5f}, {sq_res_m2/sq_full_m2*100:.2f}%)")
        print(f"Discrepancia Método 1 vs Método 2: {abs(norm_full - np.sqrt(sq_full_m2)):.2e}")
        
        # Verify Projection Identities for mixed components:
        t_eee = T3[0,0,0]
        t_ccc = T3[1,1,1]
        t_eec = T3[0,0,1]
        t_ecc = T3[0,1,1]
        
        # Theoretical linear projections:
        pred_eeh = a1 * t_eee + a2 * t_eec
        pred_ech = a1 * t_eec + a2 * t_ecc
        pred_cch = a1 * t_ecc + a2 * t_ccc
        pred_ehh = a1**2 * t_eee + 2*a1*a2 * t_eec + a2**2 * t_ecc
        pred_chh = a1**2 * t_eec + 2*a1*a2 * t_ecc + a2**2 * t_ccc
        pred_hhh = a1**3 * t_eee + 3*a1**2*a2 * t_eec + 3*a1*a2**2 * t_ecc + a2**3 * t_ccc
        
        print("\n--- Verificación de Identidades de Proyección ---")
        print(f"T_EEH^lin: Calculado = {T3_lin[0,0,2]:.4f}, Teórico a1*T_EEE + a2*T_EEC = {pred_eeh:.4f} (Diff: {abs(T3_lin[0,0,2] - pred_eeh):.2e})")
        print(f"T_ECH^lin: Calculado = {T3_lin[0,1,2]:.4f}, Teórico a1*T_EEC + a2*T_ECC = {pred_ech:.4f} (Diff: {abs(T3_lin[0,1,2] - pred_ech):.2e})")
        print(f"T_CCH^lin: Calculado = {T3_lin[1,1,2]:.4f}, Teórico a1*T_ECC + a2*T_CCC = {pred_cch:.4f} (Diff: {abs(T3_lin[1,1,2] - pred_cch):.2e})")
        print(f"T_EHH^lin: Calculado = {T3_lin[0,2,2]:.4f}, Teórico E[e*h_lin^2] = {pred_ehh:.4f} (Diff: {abs(T3_lin[0,2,2] - pred_ehh):.2e})")
        print(f"T_CHH^lin: Calculado = {T3_lin[1,2,2]:.4f}, Teórico E[c*h_lin^2] = {pred_chh:.4f} (Diff: {abs(T3_lin[1,2,2] - pred_chh):.2e})")
        print(f"T_HHH^lin: Calculado = {T3_lin[2,2,2]:.4f}, Teórico E[h_lin^3]   = {pred_hhh:.4f} (Diff: {abs(T3_lin[2,2,2] - pred_hhh):.2e})")
        
        print("\n--- TABLA CONSOLIDADA EXACTA (10 COMPONENTES) ---")
        print(f"{'Componente':<10} {'Mult':<5} {'Observado':<12} {'Proy Lineal':<12} {'Residual':<12} {'Explicado %':<12}")
        for name, i, j, k, w, _ in comps:
            obs_v = T3[i,j,k]
            lin_v = T3_lin[i,j,k]
            res_v = T3_res[i,j,k]
            if name in ['T_EEE', 'T_CCC']:
                pct_str = "Marginal"
            elif name in ['T_EEC', 'T_ECC']:
                pct_str = "Superficie"
            else:
                pct = (lin_v / obs_v) * 100.0 if abs(obs_v) > 1e-4 else np.nan
                pct_str = f"{pct:.1f}%"
            print(f"{name:<10} {w:<5} {obs_v:+10.4f}   {lin_v:+10.4f}   {res_v:+10.4f}   {pct_str:<12}")

    # =========================================================================
    # AUDITORÍA DE INDEPENDENCIA Y SIGNIFICANCIA: PERIODO DE VALIDACIÓN 2000–2018
    # =========================================================================
    print("\n" + "="*80)
    print("AUDITORÍA DE INDEPENDENCIA Y SIGNIFICANCIA: PERIODO DE VALIDACIÓN 2000–2018")
    print("="*80)
    
    # Subconjunto 2000-2018 (meses 238 a 466, N=228)
    df['time'] = pd.date_range('1980-03-01', periods=len(df), freq='MS')
    val_mask = df['time'] >= '2000-01-01'
    df_val = df[val_mask].copy()
    N_val = len(df_val)
    print(f"Periodo de validación 2000–2018: N = {N_val} meses")
    
    e_val = (df_val['Indice_E'].values - np.mean(df_val['Indice_E'].values)) / np.std(df_val['Indice_E'].values, ddof=1)
    c_val = (df_val['Indice_C'].values - np.mean(df_val['Indice_C'].values)) / np.std(df_val['Indice_C'].values, ddof=1)
    h_val = (df_val['D20_East'].interpolate().values - np.mean(df_val['D20_East'].interpolate().values)) / np.std(df_val['D20_East'].interpolate().values, ddof=1)
    
    # Compute T_ECH(+3, +2) and T_ECH(-3, -2) on validation set
    tau1, tau2 = 3, 2
    t_start = max(0, tau1, tau2)
    t_end = N_val
    t_idx = np.arange(t_start, t_end)
    t_fwd_val = np.mean(e_val[t_idx] * c_val[t_idx - tau1] * h_val[t_idx - tau2])
    
    t_start_rev = 0
    t_end_rev = min(N_val, N_val - tau1, N_val - tau2)
    t_idx_rev = np.arange(t_start_rev, t_end_rev)
    t_rev_val = np.mean(e_val[t_idx_rev] * c_val[t_idx_rev + tau1] * h_val[t_idx_rev + tau2])
    delta_val = t_fwd_val - t_rev_val
    
    print(f"T_ECH(+3, +2) [2000–2018]: {t_fwd_val:+.4f}")
    print(f"T_ECH(-3, -2) [2000–2018]: {t_rev_val:+.4f}")
    print(f"Delta T_ECH   [2000–2018]: {delta_val:+.4f}")
    
    # Moving Block Bootstrap on validation set specifically
    np.random.seed(42)
    B = 2000
    L = 18
    n_blocks = int(np.ceil(N_val / L))
    boot_deltas_val = []
    
    for _ in range(B):
        start_indices = np.random.randint(0, N_val - L + 1, size=n_blocks)
        sampled_idx = np.concatenate([np.arange(s, s + L) for s in start_indices])[:N_val]
        eb, cb, hb = e_val[sampled_idx], c_val[sampled_idx], h_val[sampled_idx]
        
        tf = np.mean(eb[t_idx] * cb[t_idx - tau1] * hb[t_idx - tau2])
        tr = np.mean(eb[t_idx_rev] * cb[t_idx_rev + tau1] * hb[t_idx_rev + tau2])
        boot_deltas_val.append(tf - tr)
        
    boot_deltas_val = np.array(boot_deltas_val)
    ci_low_val = np.percentile(boot_deltas_val, 2.5)
    ci_high_val = np.percentile(boot_deltas_val, 97.5)
    se_val = np.std(boot_deltas_val)
    print(f"Bootstrap en Bloques (2000–2018, L=18m, B=2000):")
    print(f"  Media: {np.mean(boot_deltas_val):+.4f} ± {se_val:.4f}")
    print(f"  95% CI: [{ci_low_val:+.4f}, {ci_high_val:+.4f}] (Excluye cero: {ci_low_val > 0})")
    
    # Fourier phase-randomized surrogates on validation set specifically
    np.random.seed(100)
    surr_deltas_val = []
    X_val = np.column_stack([e_val, c_val, h_val])
    fft_X = np.fft.rfft(X_val, axis=0)
    n_freq = fft_X.shape[0]
    
    for _ in range(B):
        phases = np.random.uniform(0, 2 * np.pi, size=(n_freq, 1))
        fft_surr = fft_X * np.exp(1j * phases)
        fft_surr[0] = fft_X[0].real
        if N_val % 2 == 0:
            fft_surr[-1] = fft_X[-1].real
        surr_X = np.fft.irfft(fft_surr, n=N_val, axis=0)
        es, cs, hs = surr_X[:, 0], surr_X[:, 1], surr_X[:, 2]
        
        tf = np.mean(es[t_idx] * cs[t_idx - tau1] * hs[t_idx - tau2])
        tr = np.mean(es[t_idx_rev] * cs[t_idx_rev + tau1] * hs[t_idx_rev + tau2])
        surr_deltas_val.append(tf - tr)
        
    surr_deltas_val = np.array(surr_deltas_val)
    p_val_fourier = np.mean(np.abs(surr_deltas_val) >= np.abs(delta_val))
    print(f"Surrogados de Fourier (2000–2018, B=2000):")
    print(f"  Media nula: {np.mean(surr_deltas_val):+.4f} ± {np.std(surr_deltas_val):.4f}")
    print(f"  p-valor (dos colas): p = {p_val_fourier:.4f}")
    
    # VAR(2) on validation set specifically
    np.random.seed(2026)
    var_mod_val = VAR(X_val).fit(2)
    var_deltas_val = []
    for _ in range(B):
        sim = var_mod_val.simulate_var(N_val)
        es, cs, hs = sim[:, 0], sim[:, 1], sim[:, 2]
        tf = np.mean(es[t_idx] * cs[t_idx - tau1] * hs[t_idx - tau2])
        tr = np.mean(es[t_idx_rev] * cs[t_idx_rev + tau1] * hs[t_idx_rev + tau2])
        var_deltas_val.append(tf - tr)
        
    var_deltas_val = np.array(var_deltas_val)
    p_val_var = np.mean(np.abs(var_deltas_val) >= np.abs(delta_val))
    print(f"Modelo Nulo VAR(2) (2000–2018, B=2000):")
    print(f"  Media nula: {np.mean(var_deltas_val):+.4f} ± {np.std(var_deltas_val):.4f}")
    print(f"  p-valor (dos colas): p = {p_val_var:.4f}")

if __name__ == '__main__':
    main()
