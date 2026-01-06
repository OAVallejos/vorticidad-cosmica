#!/usr/bin/env python3
"""                         
STABLE VERSION - WITHOUT OVERLY STRICT PRIORS
"""                                                                     
import numpy as np
import emcee
import time
import corner
import matplotlib.pyplot as plt
from scipy import stats
from numba import njit
import json

print("="*70)
print("📊 MCMC - DUAL MODEL (STABLE VERSION)")
print("🎯 RELAXED CONFIGURATION FOR CONVERGENCE")
print("="*70)

# ============================================================================
# 1. PHYSICAL PARAMETERS (STABLE VERSION)
# ============================================================================

A_OMEGA = 3.10e9
MC = 224.5
H0_PLANCK = 67.4
S8_PLANCK = 0.832
OMEGA_M = 0.315

# Goldilocks factor but with flexibility
S8_SUPPRESSION_FACTOR = 0.175

SH0ES_DATA = {"H0": 73.04, "sigma": 1.04}
DES_DATA = {"S8": 0.776, "sigma": 0.017}

print("\n📋 STABLE CONFIGURATION:")
print(f"   • A_ω = {A_OMEGA/1e9:.1f}×10⁹")
print(f"   • Suppression factor = {S8_SUPPRESSION_FACTOR}")
print(f"   • H₀ Planck = {H0_PLANCK}, S₈ Planck = {S8_PLANCK}")

# ============================================================================
# 2. SIMPLIFIED MODEL (MORE STABLE)
# ============================================================================

@njit
def simple_dual_model(params):
    """Simplified model without excessive controls"""
    ln_A_S8, ln_gamma = params

    # Convert to physical values
    A_S8 = np.exp(ln_A_S8)
    A_H0 = A_OMEGA - A_S8

    # Fractions
    f_S8 = A_S8 / A_OMEGA
    f_H0 = A_H0 / A_OMEGA

    # S₈ with suppression
    S8_pred = S8_PLANCK * (1 - S8_SUPPRESSION_FACTOR * f_S8)

    # H₀ with simple correction
    # γ controls how much the H₀ fraction affects the correction
    gamma = np.exp(ln_gamma)
    correction = f_H0 * 0.15 * gamma  # 15% max × gamma factor

    H0_pred = H0_PLANCK * (1 + correction)

    return H0_pred, S8_pred, A_S8, A_H0, gamma, f_S8

# ============================================================================
# 3. STABLE LIKELIHOOD (WITHOUT HARD PENALTIES)
# ============================================================================

def stable_likelihood(params):
    """Stable likelihood with soft priors"""

    # Calculate predictions
    H0_pred, S8_pred, A_S8, A_H0, gamma, f_S8 = simple_dual_model(params)

    # Verify basic physical values
    if (H0_pred < 60 or H0_pred > 90 or
        S8_pred < 0.6 or S8_pred > 0.9 or
        np.isnan(H0_pred) or np.isnan(S8_pred)):
        return -1e10

    # χ² with data
    chi2_H0 = ((H0_pred - SH0ES_DATA["H0"]) / SH0ES_DATA["sigma"])**2
    chi2_S8 = ((S8_pred - DES_DATA["S8"]) / DES_DATA["sigma"])**2

    # Soft priors (not hard walls)
    prior_f_S8 = -0.5 * ((f_S8 - 0.70) / 0.15)**2  # Centered at 70%, width ±15%
    prior_gamma = -0.5 * ((gamma - 2.4) / 0.5)**2  # Centered at 2.4, width ±0.5

    # Soft penalty if S₈ is too low (but not a wall)
    if S8_pred < 0.74:
        S8_penalty = -10.0 * (0.74 - S8_pred)**2  # Soft, not hard
    else:
        S8_penalty = 0.0

    logL = -0.5 * (chi2_H0 + chi2_S8) + prior_f_S8 + prior_gamma + S8_penalty

    return logL

def lcdm_likelihood():
    """ΛCDM baseline"""
    chi2_H0 = ((H0_PLANCK - SH0ES_DATA["H0"]) / SH0ES_DATA["sigma"])**2
    chi2_S8 = ((S8_PLANCK - DES_DATA["S8"]) / DES_DATA["sigma"])**2
    return -0.5 * (chi2_H0 + chi2_S8)

# ============================================================================
# 4. ROBUST MCMC
# ============================================================================

def run_robust_mcmc():
    """MCMC with robust configuration"""

    print("\n" + "="*70)
    print("🚀 RUNNING ROBUST MCMC")
    print("="*70)

    ndim, nwalkers = 2, 48  # More walkers for better exploration
    nsteps, nburn = 3000, 1000

    # Diversified initial points
    np.random.seed(42)
    p0 = np.zeros((nwalkers, ndim))

    # Diversify walkers around different points
    for i in range(nwalkers):
        if i % 4 == 0:
            # Center: 70/30 split, γ=2.4
            p0[i] = [np.log(0.70 * A_OMEGA), np.log(2.4)]
        elif i % 4 == 1:
            # More towards S₈: 75/25 split
            p0[i] = [np.log(0.75 * A_OMEGA), np.log(2.2)]
        elif i % 4 == 2:
            # More towards H₀: 65/35 split
            p0[i] = [np.log(0.65 * A_OMEGA), np.log(2.6)]
        else:
            # Exploratory
            p0[i] = [np.log(0.68 * A_OMEGA + 0.1e9 * np.random.randn()),
                     np.log(2.4 + 0.3 * np.random.randn())]

    print(f"📋 ROBUST CONFIGURATION:")
    print(f"   • Walkers: {nwalkers} (diversified)")
    print(f"   • Steps: {nsteps}, Burn: {nburn}")
    print(f"   • Soft priors, no hard walls")

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        stable_likelihood,
        moves=emcee.moves.StretchMove(a=2.5)  # Larger step
    )

    # Burn-in
    print("\n🔥 Burn-in...")
    state = sampler.run_mcmc(p0, nburn, progress=True)

    # Reset and main MCMC
    sampler.reset()
    print("\n🔄 Main MCMC...")
    start = time.time()
    sampler.run_mcmc(state, nsteps, progress=True)
    print(f"✅ Completed in {time.time()-start:.1f}s")

    return sampler

# ============================================================================
# 5. ANALYSIS WITH PROTECTION AGAINST EMPTY ARRAYS
# ============================================================================

def safe_analyze_results(sampler):
    """Analysis with protection against empty samples"""

    # Get samples (without thinning first)
    samples_raw = sampler.get_chain(discard=500, flat=True)
    log_prob_raw = sampler.get_log_prob(discard=500, flat=True)

    print(f"\n📈 RAW STATISTICS:")
    print(f"   • Total samples: {len(samples_raw)}")
    print(f"   • Acceptance rate: {sampler.acceptance_fraction.mean():.3f}")

    # Filter only samples with finite likelihood
    valid_idx = np.isfinite(log_prob_raw)
    samples = samples_raw[valid_idx]
    log_prob = log_prob_raw[valid_idx]

    print(f"   • Valid samples: {len(samples)}")

    if len(samples) == 0:
        print("❌ ERROR: No valid samples. Using default values.")
        return None

    # Best sample
    idx_best = np.argmax(log_prob)
    params_best = samples[idx_best]

    H0_best, S8_best, A_S8_best, A_H0_best, gamma_best, f_S8_best = simple_dual_model(params_best)

    # Calculate predictions for all valid samples
    H0_vals = []
    S8_vals = []

    for i in range(min(1000, len(samples))):
        params = samples[i]
        H0_pred, S8_pred, _, _, _, _ = simple_dual_model(params)
        if np.isfinite(H0_pred) and np.isfinite(S8_pred):
            H0_vals.append(H0_pred)
            S8_vals.append(S8_pred)

    if len(H0_vals) == 0:
        print("❌ ERROR: No valid predictions.")
        return None

    H0_vals = np.array(H0_vals)
    S8_vals = np.array(S8_vals)

    # Robust statistics
    H0_median = np.median(H0_vals)
    H0_std = np.std(H0_vals)
    S8_median = np.median(S8_vals)
    S8_std = np.std(S8_vals)

    # Tensions
    tension_H0 = abs(H0_median - SH0ES_DATA["H0"]) / np.sqrt(H0_std**2 + SH0ES_DATA["sigma"]**2)
    tension_S8 = abs(S8_median - DES_DATA["S8"]) / np.sqrt(S8_std**2 + DES_DATA["sigma"]**2)

    # Metrics
    chi2_dual = ((H0_median - SH0ES_DATA["H0"])/SH0ES_DATA["sigma"])**2 + \
                ((S8_median - DES_DATA["S8"])/DES_DATA["sigma"])**2

    chi2_lcdm = ((H0_PLANCK - SH0ES_DATA["H0"])/SH0ES_DATA["sigma"])**2 + \
                ((S8_PLANCK - DES_DATA["S8"])/DES_DATA["sigma"])**2

    logL_dual = np.max(log_prob)
    logL_lcdm = lcdm_likelihood()

    # Bayes Factor Correction
    ln_B = logL_dual - logL_lcdm  # This is the correct form!
    B = np.exp(ln_B) if np.isfinite(ln_B) else 0.0

    # Significance
    delta_chi2 = chi2_lcdm - chi2_dual
    p_value = 1 - stats.chi2.cdf(delta_chi2, 2) if delta_chi2 > 0 else 1.0
    sigma_eq = stats.norm.ppf(1 - p_value/2) if p_value > 0 else 0.0

    results = {
        'params_best': params_best,
        'predictions': {
            'H0_median': H0_median,
            'H0_std': H0_std,
            'S8_median': S8_median,
            'S8_std': S8_std,
            'f_S8': f_S8_best,
            'f_H0': 1 - f_S8_best,
            'gamma': gamma_best
        },
        'tensions': {
            'H0': tension_H0,
            'S8': tension_S8,
            'H0_lcdm': abs(H0_PLANCK - SH0ES_DATA["H0"])/SH0ES_DATA["sigma"],
            'S8_lcdm': abs(S8_PLANCK - DES_DATA["S8"])/DES_DATA["sigma"]
        },
        'metrics': {
            'chi2_lcdm': chi2_lcdm,
            'chi2_dual': chi2_dual,
            'delta_chi2': delta_chi2,
            'ln_B': ln_B,
            'B': B,
            'p_value': p_value,
            'sigma_eq': sigma_eq,
            'logL_dual': logL_dual,
            'logL_lcdm': logL_lcdm
        },
        'samples': {
            'H0': H0_vals,
            'S8': S8_vals,
            'samples': samples,
            'log_prob': log_prob
        }
    }

    return results, sampler

# ============================================================================
# 6. SAFE PLOTS
# ============================================================================



def create_safe_plots(results, sampler):
    """Plots with error protection"""

    try:
        print("\n🎨 Generating plots...")

        # 1. Corner plot only if there are enough samples
        if len(results['samples']['samples']) > 100:
            fig1 = plt.figure(figsize=(10, 10))

            # Convert to physical parameters
            samples_phys = np.zeros_like(results['samples']['samples'])
            samples_phys[:, 0] = np.exp(results['samples']['samples'][:, 0]) / 1e9
            samples_phys[:, 1] = np.exp(results['samples']['samples'][:, 1])

            corner.corner(
                samples_phys[:1000],  # Use only 1000 for speed
                labels=[r"$A_{S_8}\ (10^9)$", r"$\gamma$"],
                show_titles=True,
                fig=fig1
            )

            plt.savefig('CORNER_PLOT_STABLE.png', dpi=300, bbox_inches='tight')
            plt.close(fig1)
            print("   • Corner plot saved")

        # 2. Prediction Histograms
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # H₀
        ax1.hist(results['samples']['H0'], bins=30, alpha=0.7, color='blue', density=True)
        ax1.axvline(SH0ES_DATA["H0"], color='red', ls='--', label='SH0ES')
        ax1.axvline(H0_PLANCK, color='green', ls='--', label='Planck')
        ax1.axvline(results['predictions']['H0_median'], color='black', label='Model')
        ax1.set_xlabel(r'$H_0$ [km/s/Mpc]')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.set_title(f'H₀ = {results["predictions"]["H0_median"]:.2f} ± {results["predictions"]["H0_std"]:.2f}')

        # S₈
        ax2.hist(results['samples']['S8'], bins=30, alpha=0.7, color='orange', density=True)
        ax2.axvline(DES_DATA["S8"], color='red', ls='--', label='DES')
        ax2.axvline(S8_PLANCK, color='green', ls='--', label='Planck')
        ax2.axvline(results['predictions']['S8_median'], color='black', label='Model')
        ax2.set_xlabel(r'$S_8$')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.set_title(f'S₈ = {results["predictions"]["S8_median"]:.3f} ± {results["predictions"]["S8_std"]:.3f}')

        plt.tight_layout()
        plt.savefig('STABLE_PREDICTIONS.png', dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print("   • Histograms saved")

        return True

    except Exception as e:
        print(f"   • Error in plotting: {e}")
        return False

# ============================================================================
# 7. MAIN FUNCTION
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🎯 FINAL ANALYSIS - STABLE CONFIGURATION")
    print("="*70)

    # Run robust MCMC
    sampler = run_robust_mcmc()

    # Analyze results with protection
    results, sampler = safe_analyze_results(sampler)

    if results is None:
        print("❌ Could not analyze results.")
        return

    # Show results
    print(f"\n🎯 STABLE RESULTS:")
    print(f"   • H₀ = {results['predictions']['H0_median']:.2f} ± {results['predictions']['H0_std']:.2f} km/s/Mpc")
    print(f"   • S₈ = {results['predictions']['S8_median']:.3f} ± {results['predictions']['S8_std']:.3f}")
    print(f"   • S₈/H₀ Fraction: {results['predictions']['f_S8']*100:.1f}% / {results['predictions']['f_H0']*100:.1f}%")
    print(f"   • γ = {results['predictions']['gamma']:.2f}")

    print(f"\n⚡ TENSIONS:")
    print(f"   • H₀: {results['tensions']['H0_lcdm']:.1f}σ → {results['tensions']['H0']:.2f}σ")
    print(f"   • S₈: {results['tensions']['S8_lcdm']:.1f}σ → {results['tensions']['S8']:.2f}σ")
    print(f"   • H₀ Reduction: {results['tensions']['H0_lcdm'] - results['tensions']['H0']:.1f}σ")
    print(f"   • S₈ Reduction: {results['tensions']['S8_lcdm'] - results['tensions']['S8']:.1f}σ")

    print(f"\n📈 STATISTICAL METRICS:")
    print(f"   • Δχ² = {results['metrics']['delta_chi2']:.1f}")
    print(f"   • ln(B) = {results['metrics']['ln_B']:.1f} (B ≈ {results['metrics']['B']:.0f}:1)")
    print(f"   • p-value = {results['metrics']['p_value']:.2e}")
    if results['metrics']['sigma_eq'] > 0:
        print(f"   • Significance = {results['metrics']['sigma_eq']:.1f}σ")

    # Create plots
    create_safe_plots(results, sampler)

    # Export results
    export_results = {
        'config': {
            'suppression_factor': S8_SUPPRESSION_FACTOR,
            'A_omega': float(A_OMEGA)
        },
        'predictions': {
            'H0': {
                'value': float(results['predictions']['H0_median']),
                'error': float(results['predictions']['H0_std']),
                'SH0ES_tension': float(results['tensions']['H0'])
            },
            'S8': {
                'value': float(results['predictions']['S8_median']),
                'error': float(results['predictions']['S8_std']),
                'DES_tension': float(results['tensions']['S8'])
            },
            'parameters': {
                'S8_fraction': float(results['predictions']['f_S8']),
                'H0_fraction': float(results['predictions']['f_H0']),
                'gamma': float(results['predictions']['gamma'])
            }
        },
        'original_tensions': {
            'H0_LCDM': float(results['tensions']['H0_lcdm']),
            'S8_LCDM': float(results['tensions']['S8_lcdm'])
        },
        'statistics': {
            'chi2_LCDM': float(results['metrics']['chi2_lcdm']),
            'chi2_Dual': float(results['metrics']['chi2_dual']),
            'delta_chi2': float(results['metrics']['delta_chi2']),
            'ln_Bayes_factor': float(results['metrics']['ln_B']),
            'p_value': float(results['metrics']['p_value'])
        }
    }

    with open('STABLE_RESULTS.json', 'w') as f:
        json.dump(export_results, f, indent=2)

    print("\n💾 Results saved: STABLE_RESULTS.json")

    # Final Table
    print("\n" + "="*70)
    print("📋 FINAL TABLE")
    print("="*70)

    print(f"""
Parameter      | ΛCDM       | Dual Model  | Improvement
---------------|------------|-------------|------------
H₀ [km/s/Mpc]  | 67.4 ± 0.5 | {results['predictions']['H0_median']:.2f} ± {results['predictions']['H0_std']:.2f} | --
S₈             | 0.832 ± 0.013 | {results['predictions']['S8_median']:.3f} ± {results['predictions']['S8_std']:.3f} | --
H₀ Tension (σ) | {results['tensions']['H0_lcdm']:.1f}        | {results['tensions']['H0']:.2f}        | {results['tensions']['H0_lcdm'] - results['tensions']['H0']:.1f}σ
S₈ Tension (σ) | {results['tensions']['S8_lcdm']:.1f}        | {results['tensions']['S8']:.2f}        | {results['tensions']['S8_lcdm'] - results['tensions']['S8']:.1f}σ
Δχ²            | --         | --          | {results['metrics']['delta_chi2']:.1f}
ln(B)          | --         | --          | {results['metrics']['ln_B']:.1f}
p-value        | --         | --          | {results['metrics']['p_value']:.2e}
    """)

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*70)

# ============================================================================
# 8. EXECUTION
# ============================================================================

if __name__ == "__main__":
    main()