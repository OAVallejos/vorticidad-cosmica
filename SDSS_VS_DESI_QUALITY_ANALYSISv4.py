#!/usr/bin/env python3      
"""
EXTENDED ROBUSTNESS ANALYSIS - DIFFERENTIATED REDSHIFT BINS
- SDSS: z=0.1-0.2 and z=0.7-0.8 (works well)
- DESI: z=0.4-0.6 and z=0.8-1.0 (high statistical power)

"""

import numpy as np
import json
from scipy import stats
from astropy.table import Table
import sys
from datetime import datetime

try:
    # We assume the Rust module is available for the user
    from cosmic_vorticity import calcular_bispectro_triangular
    print("✅ Rust module loaded for fast bispectrum calculation.")
except ImportError:
    print("❌ Rust module not available. Aborting.")
    sys.exit()

print("🎯 EXTENDED ROBUSTNESS ANALYSIS - OPTIMIZED REDSHIFT BINS (V5 - Corrected Power)")
print("=============================================================================")

# --- DATASET-SPECIFIC CONFIGURATION ---
# SDSS Strategy: Natural percentiles with VDISP > 150 km/s
SDSS_Q33 = 207.9
SDSS_Q66 = 262.2
SDSS_CUTS = (SDSS_Q33, SDSS_Q66)

# DESI Strategy: Optimized physical cuts
DESI_Q_LOW = 200.0
DESI_Q_HIGH = 250.0
DESI_CUTS = (DESI_Q_LOW, DESI_Q_HIGH)

# --- COMMON CONFIGURATION ---
VDISP_MIN_CALIDAD = 150.0  # Common quality filter
H0_TEST = 1.1
SAMPLE_SIZE = 500
N_VALIDATION_SAMPLES = 80
configs_escalenas = [(1, 2, 3), (1, 3, 4), (2, 3, 5), (1, 4, 5), (2, 4, 6), (3, 4, 7)]
l_max = 8

# --- DIFFERENTIATED REDSHIFT BINS ---
# SDSS: Works well with original bins
bins_redshift_sdss = [('z01_02', 0.1, 0.2), ('z07_08', 0.7, 0.8)]

# DESI: Bins optimized for high statistical power
bins_redshift_desi = [('z04_06', 0.4, 0.6), ('z08_10', 0.8, 1.0)]

def analisis_potencia_estadistica(resultados, efecto_objetivo=0.1):
    """Calculates Power (1-beta) to detect effect delta=0.1 (10%) using real SEM."""
    print("\n🔬 STATISTICAL POWER ANALYSIS (vs 10% effect)")
    print("--------------------------------------------------")

    for grupo, datos in resultados.items():

        # --- CORRECTION: USE REAL SEM (evolution_sem) ---
        if 'evolution_sem' not in datos or datos['evolution_sem'] == 0:
            # Fallback for groups with invalid results
            print(f"   - {grupo:<20}: Power not calculable (SEM not available or zero)")
            continue

        sem_real = datos['evolution_sem']
        # --- END CORRECTION ---

        Z_crit_upper = stats.norm.ppf(0.975) # Z-score for 95% confidence

        # Calculate critical values (X_crit) for H0 test
        X_crit_upper = H0_TEST + Z_crit_upper * sem_real
        X_crit_lower = H0_TEST - Z_crit_upper * sem_real

        # Define alternative hypotheses
        Mu_alt_pos = H0_TEST + efecto_objetivo
        Mu_alt_neg = H0_TEST - efecto_objetivo

        # Calculate power for positive alternative hypothesis (Mu_alt_pos)
        Z_alt_upper_pos = (X_crit_upper - Mu_alt_pos) / sem_real
        Z_alt_lower_pos = (X_crit_lower - Mu_alt_pos) / sem_real
        # Power = probability of rejecting H0 when Mu_alt_pos is true
        power_pos = stats.norm.cdf(Z_alt_lower_pos) + (1 - stats.norm.cdf(Z_alt_upper_pos))

        # Calculate power for negative alternative hypothesis (Mu_alt_neg)
        Z_alt_upper_neg = (X_crit_upper - Mu_alt_neg) / sem_real
        Z_alt_lower_neg = (X_crit_lower - Mu_alt_neg) / sem_real
        # Power = probability of rejecting H0 when Mu_alt_neg is true
        power_neg = stats.norm.cdf(Z_alt_lower_neg) + (1 - stats.norm.cdf(Z_alt_upper_neg))

        # Average power for a two-tailed test
        potencia = (power_pos + power_neg) / 2

        print(f"   - {grupo:<20}: Power to detect 10% = {potencia:.3f}")

def ejecutar_analisis_grupos(vdisp_full, redshift_full, mass_groups, dataset_name, bins_redshift):
    """Runs analysis for specific mass groups with custom redshift bins."""
    resultados = {}

    for group_name, (vdisp_min, vdisp_max, label_masa) in mass_groups.items():
        print(f"\n--- {dataset_name}: {label_masa} ({vdisp_min:.1f} - {vdisp_max:.1f} km/s) ---")

        # Group statistics
        mask_grupo = (vdisp_full >= vdisp_min) & (vdisp_full < vdisp_max)
        n_galaxias = np.sum(mask_grupo)
        print(f"    📊 Galaxies in group: {n_galaxias:,}")

        galaxies_z_low = None
        galaxies_z_high = None

        for label_z, z_min, z_max in bins_redshift:
            mask = (redshift_full >= z_min) & (redshift_full < z_max) & mask_grupo
            galaxies_bin = vdisp_full[mask]

            if len(galaxies_bin) < SAMPLE_SIZE:
                print(f"    ⚠️  {label_z}: Only {len(galaxies_bin):,} galaxies ({SAMPLE_SIZE} needed)")
                continue

            # Identify low and high bins based on dataset
            if dataset_name == "SDSS":
                if 'z01_02' in label_z:
                    galaxies_z_low = galaxies_bin
                    print(f"    🔵 z={z_min}-{z_max}: {len(galaxies_bin):,} galaxies")
                elif 'z07_08' in label_z:
                    galaxies_z_high = galaxies_bin
                    print(f"    🔴 z={z_min}-{z_max}: {len(galaxies_bin):,} galaxies")
            else:  # DESI
                if 'z04_06' in label_z:
                    galaxies_z_low = galaxies_bin
                    print(f"    🔵 z={z_min}-{z_max}: {len(galaxies_bin):,} galaxies")
                elif 'z08_10' in label_z:
                    galaxies_z_high = galaxies_bin
                    print(f"    🔴 z={z_min}-{z_max}: {len(galaxies_bin):,} galaxies")

        # EVOLUTION CALCULATION
        if galaxies_z_low is not None and galaxies_z_high is not None:
            n_samples_max = min(N_VALIDATION_SAMPLES,
                                len(galaxies_z_low) // SAMPLE_SIZE,
                                len(galaxies_z_high) // SAMPLE_SIZE)

            if n_samples_max > 0:
                print(f"    🔄 Generating {n_samples_max} samples...")

                indices_low = np.random.permutation(len(galaxies_z_low))
                indices_high = np.random.permutation(len(galaxies_z_high))
                evolutions = []

                for i in range(n_samples_max):
                    start, end = i * SAMPLE_SIZE, (i + 1) * SAMPLE_SIZE
                    sample_low = galaxies_z_low[indices_low[start:end]].tolist()
                    sample_high = galaxies_z_high[indices_high[start:end]].tolist()

                    bispectra_low = calcular_bispectro_triangular(sample_low, l_max, configs_escalenas)
                    bispectra_high = calcular_bispectro_triangular(sample_high, l_max, configs_escalenas)

                    if bispectra_high and bispectra_low:
                        avg_high = np.mean([abs(b) for b in bispectra_high])
                        avg_low = np.mean([abs(b) for b in bispectra_low])
                        evol_esc = avg_high / avg_low if avg_low > 0 else np.nan
                        evolutions.append(evol_esc)

                # STATISTICAL ANALYSIS
                evolutions = np.array(evolutions)
                evolutions = evolutions[~np.isnan(evolutions)]

                if len(evolutions) > 1:
                    obs_mean = np.mean(evolutions)
                    obs_std = np.std(evolutions, ddof=1)
                    n_obs = len(evolutions)
                    obs_sem = obs_std / np.sqrt(n_obs) # <-- THIS IS THE REAL SEM

                    t = abs(obs_mean - H0_TEST) / obs_sem
                    p = 2 * (1 - stats.t.cdf(t, n_obs-1))
                    sigma = stats.norm.ppf(1 - p/2)
                    if np.isinf(sigma): sigma = t

                    resultados[label_masa] = {
                        'evolution_mean': float(obs_mean),
                        'significance_11': float(sigma),
                        'evolution_std': float(obs_std),
                        'evolution_sem': float(obs_sem), # <-- ADDED FOR POWER CORRECTION
                        'N_samples': n_obs,
                        'vdisp_rango': f"{vdisp_min:.1f}-{vdisp_max:.1f}",
                        'n_galaxias_grupo': int(n_galaxias)
                    }

                    print(f"    📈 RESULTS:")
                    print(f"        • Mean Evolution: {obs_mean:.3f}×")
                    print(f"        • Significance vs {H0_TEST}x: {sigma:.2f}σ")
                    print(f"        • SEM (real): {obs_sem:.4f}")
                    print(f"        • Valid samples: {n_obs}")

                    if sigma >= 5.0:
                        print(f"        🎉 **SOLID EVIDENCE (>5σ)**")
                else:
                    print("    ❌ Statistical analysis not available.")
            else:
                print("     ❌ Could not obtain sufficient samples.")
        else:
            print("    ❌ Not enough galaxies in both redshift bins")

    return resultados

def ejecutar_analisis_dual(dataset_name, vdisp_data, redshift_data, cuts, estrategia, bins_redshift):
    """Runs analysis with dataset-specific strategy and custom redshift bins."""
    print(f"\n" + "="*80)
    print(f"🎯 RUNNING {dataset_name} ANALYSIS - STRATEGY: {estrategia}")
    print("="*80)

    q_low, q_high = cuts

    # Apply common quality filter
    mask_calidad = vdisp_data >= VDISP_MIN_CALIDAD
    vdisp_filtrado = vdisp_data[mask_calidad]
    redshift_filtrado = redshift_data[mask_calidad]

    print(f"🔧 {dataset_name} CONFIGURATION:")
    print(f"    • Strategy: {estrategia}")
    print(f"    • Cuts: LOW={q_low:.1f} | HIGH={q_high:.1f} km/s")
    print(f"    • Redshift bins: {bins_redshift}")
    print(f"    • VDISP filter > {VDISP_MIN_CALIDAD} km/s: {len(vdisp_filtrado):,} galaxies")

    # Define mass groups
    mass_groups = {
        "VDISP_LOW": (VDISP_MIN_CALIDAD, q_low, f'LOW_MASS_{dataset_name}'),
        "VDISP_MID": (q_low, q_high, f'MID_MASS_{dataset_name}'),
        "VDISP_HIGH": (q_high, 1000.0, f'HIGH_MASS_{dataset_name}')
    }

    # Run analysis
    resultados = ejecutar_analisis_grupos(vdisp_filtrado, redshift_filtrado, mass_groups, dataset_name, bins_redshift)

    return resultados

def guardar_resultados_completos(resultados_sdss, resultados_desi, metadata):
    """Saves results in JSON format."""
    output = {
        'metadata': metadata,
        'configuration': {
            'common_filter': f'VDISP > {VDISP_MIN_CALIDAD} km/s',
            'sdss_cuts': {'low': SDSS_Q33, 'high': SDSS_Q66, 'strategy': 'Natural Percentiles'},
            'desi_cuts': {'low': DESI_Q_LOW, 'high': DESI_Q_HIGH, 'strategy': 'Optimized Physical Cuts'},
            'sdss_redshift_bins': bins_redshift_sdss,
            'desi_redshift_bins': bins_redshift_desi,
            'H0_TEST': H0_TEST,
            'n_samples': N_VALIDATION_SAMPLES,
            'sample_size': SAMPLE_SIZE
        },
        'sdss_results': resultados_sdss,
        'desi_results': resultados_desi,
        'timestamp': datetime.now().isoformat()
    }

    filename = 'EXTENDED_ROBUSTNESS_SDSS_DESI_REDSHIFT_V5.json'
    try:
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Results saved in '{filename}'.")
    except IOError:
        print("\n❌ Error saving JSON file.")

# --- MAIN CODE ---
def main():
    print("📥 LOADING DATASETS...")


    # Load SDSS
    try:
        data_sdss = np.load('sdss_vdisp_calidad.npz')
        vdisp_sdss = data_sdss['VDISP']
        redshift_sdss = data_sdss['Z']
        print(f"✅ SDSS loaded: {len(vdisp_sdss):,} galaxies")
    except FileNotFoundError:
        print("❌ Error: 'sdss_vdisp_calidad.npz' not found.")
        return

    # Load DESI
    try:
        tabla_desi = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
        vdisp_desi = np.array(tabla_desi['VDISP'])
        redshift_desi = np.array(tabla_desi['Z'])
        print(f"✅ DESI loaded: {len(vdisp_desi):,} galaxies")
    except FileNotFoundError:
        print("❌ Error: 'DATASET_LRG_VDISP_FLUXR_FINAL.fits' not found.")
        return

    # RUN DUAL ANALYSIS WITH DIFFERENTIATED BINS
    resultados_sdss = ejecutar_analisis_dual("SDSS", vdisp_sdss, redshift_sdss,
                                           SDSS_CUTS, "Natural Percentiles", bins_redshift_sdss)

    resultados_desi = ejecutar_analisis_dual("DESI", vdisp_desi, redshift_desi,
                                           DESI_CUTS, "Optimized Physical Cuts", bins_redshift_desi)

    # COMPARATIVE SUMMARY
    print("\n" + "="*80)
    print("📊 COMPARATIVE SUMMARY: SDSS vs DESI (OPTIMIZED BINS)")
    print("="*80)

    print("\n📈 SDSS - Natural Percentiles (z=0.1-0.2, z=0.7-0.8):")
    for grupo, res in resultados_sdss.items():
        print(f"   {grupo:>20}: {res['evolution_mean']:6.3f}× | {res['significance_11']:5.2f}σ | {res['n_galaxias_grupo']:>8,} galaxies")

    print("\n📈 DESI - Optimized Physical Cuts (z=0.4-0.6, z=0.8-1.0):")
    for grupo, res in resultados_desi.items():
        print(f"   {grupo:>20}: {res['evolution_mean']:6.3f}× | {res['significance_11']:5.2f}σ | {res['n_galaxias_grupo']:>8,} galaxies")

    # CORRECTED POWER ANALYSIS
    if resultados_sdss:
        analisis_potencia_estadistica(resultados_sdss)
    if resultados_desi:
        analisis_potencia_estadistica(resultados_desi)

    # SAVE RESULTS
    metadata_final = {
        'objective': 'Extended robustness analysis with differentiated redshift bins',
        'key_finding': 'Comparison SDSS (traditional bins) vs DESI (optimized bins)',
        'interpretation': 'Optimized bins allow full analysis of all mass groups in DESI',
        'important_note': 'Statistical Power calculation has been corrected to use real Standard Error of the Mean (SEM).'
    }

    guardar_resultados_completos(resultados_sdss, resultados_desi, metadata_final)

    print(f"\n✅ EXTENDED ROBUSTNESS ANALYSIS COMPLETED (CORRECTED POWER).")

if __name__ == "__main__":
    main()