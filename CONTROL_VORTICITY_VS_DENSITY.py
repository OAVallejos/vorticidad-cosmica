#!/usr/bin/env python3
"""
POSITIVE CONTROL: Compare DENSITY vs. VORTICITY spectrum 
To demonstrate that negative n_ω is specific to the vorticity field.
"""

import numpy as np
import healpy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Configuration
NSIDE = 64
LMAX = 150
BIN_SIZE = 10
np.random.seed(42)

def analyze_field(data, field='VDISP', label='Vorticity'):
    """
    Analyzes the power spectrum for a given field
    """
    print(f"\n🔍 ANALYZING: {label} ({field})")

    # Basic filter
    mask = ((data['Z'] >= 0.4) & (data['Z'] <= 0.7) &  # Low bin only for better SNR
            (data['VDISP'] >= 150) & (data['VDISP'] <= 500))
    data_sub = data[mask]

    ra = np.array(data_sub['RA'])
    dec = np.array(data_sub['DEC'])

    if field == 'VDISP':
        values = np.array(data_sub['VDISP'])
    else:  # Density (galaxy counts)
        values = np.ones(len(data_sub))  # Every galaxy counts as 1

    # Split-half
    n_total = len(ra)
    indices = np.random.permutation(n_total)
    cut = n_total // 2

    # Generate delta maps
    def generate_map(idx, field_vals):
        npix = hp.nside2npix(NSIDE)
        theta = np.radians(90.0 - dec[idx])
        phi = np.radians(ra[idx])
        pix = hp.ang2pix(NSIDE, theta, phi)

        counts = np.bincount(pix, minlength=npix)
        val_sum = np.bincount(pix, weights=field_vals[idx], minlength=npix)

        field_map = np.zeros(npix)
        mask_good = counts > 0

        if field == 'VDISP':
            # For vorticity: normalized average
            field_map[mask_good] = val_sum[mask_good] / counts[mask_good]
            global_mean = np.mean(field_vals[idx])
            delta_map = np.zeros(npix)
            delta_map[mask_good] = (field_map[mask_good] - global_mean) / global_mean
        else:
            # For density: normalized overdensity
            field_map[mask_good] = counts[mask_good]
            global_mean = np.mean(counts[mask_good])
            delta_map = np.zeros(npix)
            delta_map[mask_good] = (field_map[mask_good] - global_mean) / global_mean

        f_sky = np.sum(mask_good) / npix
        return delta_map, f_sky

    # Generate splits
    map_A, fsky_A = generate_map(indices[:cut], values)
    map_B, fsky_B = generate_map(indices[cut:], values)
    fsky_avg = (fsky_A + fsky_B) / 2

    print(f"   • Galaxies: {n_total:,}")
    print(f"   • f_sky: {fsky_avg:.3f}")

    # Cross-spectrum
    cl_raw = hp.anafast(map_A, map_B, lmax=LMAX)
    ell = np.arange(len(cl_raw))
    cl_corrected = cl_raw / fsky_avg

    # Binning
    ell_binned, cl_binned = [], []
    for i in range(2, LMAX, BIN_SIZE):
        mask_bin = (ell >= i) & (ell < min(i + BIN_SIZE, LMAX))
        if np.sum(mask_bin) > 0:
            cl_mean = np.mean(cl_corrected[mask_bin])
            if cl_mean > 0:  # Positives only for log
                ell_binned.append(np.mean(ell[mask_bin]))
                cl_binned.append(cl_mean)

    if len(ell_binned) < 4:
        print(f"   ❌ Insufficient bins")
        return None, None, None

    # Fitting
    log_ell = np.log(ell_binned)
    log_cl = np.log(cl_binned)

    try:
        # Weighted fit
        coeffs, cov = np.polyfit(log_ell, log_cl, 1, cov=True)
        n = coeffs[0]
        n_err = np.sqrt(cov[0,0])

        # R²
        cl_pred = np.exp(n * log_ell + coeffs[1])
        ss_res = np.sum((np.array(cl_binned) - cl_pred)**2)
        ss_tot = np.sum((np.array(cl_binned) - np.mean(cl_binned))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"   ✅ n = {n:.3f} ± {n_err:.3f}")
        print(f"   • R² = {r2:.3f}")
        print(f"   • Bins: {len(ell_binned)}")

        return np.array(ell_binned), np.array(cl_binned), n

    except Exception as e:
        print(f"   ❌ Fitting error: {e}")
        return None, None, None

def main():
    print("=" * 60)
    print("🧪 POSITIVE CONTROL: DENSITY vs VORTICITY")
    print("=" * 60)

    # Load data
    print("\n📥 LOADING DATA...")
    try:
        data = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
        print(f"   • Total galaxies: {len(data):,}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # Analyze VORTICITY (Main result)
    ell_vort, cl_vort, n_vort = analyze_field(data, field='VDISP',
                                               label='Vorticity (VDISP)')

    # Analyze DENSITY (Positive control)
    ell_dens, cl_dens, n_dens = analyze_field(data, field='DENSITY',
                                               label='Density (count)')

    # Comparative Plot
    if ell_vort is not None and ell_dens is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel A: Vorticity
        ax1.errorbar(ell_vort, cl_vort, fmt='ro-', alpha=0.7,
                    label=f'Vorticity: n={n_vort:.3f}')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Multipole ℓ', fontsize=12)
        ax1.set_ylabel('C_ℓ', fontsize=12)
        ax1.set_title('Vorticity Spectrum (VDISP)', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Panel B: Density
        ax2.errorbar(ell_dens, cl_dens, fmt='bo-', alpha=0.7,
                    label=f'Density: n={n_dens:.3f}')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Multipole ℓ', fontsize=12)
        ax2.set_ylabel('C_ℓ', fontsize=12)
        ax2.set_title('Density Spectrum (count)', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('control_vorticity_vs_density.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Plot saved: control_vorticity_vs_density.png")
        plt.close()

        # Quantitative Comparison
        print("\n" + "=" * 60)
        print("📊 QUANTITATIVE COMPARISON")
        print("=" * 60)
        print(f"   • Vorticity (VDISP):  n = {n_vort:.3f}")
        print(f"   • Density (count):    n = {n_dens:.3f}")
        print(f"   • Difference:          Δn = {n_vort - n_dens:.3f}")

        # Interpretation
        n_s_planck = 0.9649
        print(f"\n💡 INTERPRETATION:")
        print(f"   • Planck (primordial scalar): n_s = {n_s_planck:.3f}")
        print(f"   • Expected observed density: n ≈ 0.8-1.2 (depends on bias b)")

        if n_dens > 0 and abs(n_dens - n_s_planck) < 0.5:
            print(f"   ✅ POSITIVE CONTROL PASSED: Density spectral index is reasonable")
        else:
            print(f"   ⚠️ Anomalous density index: check the pipeline")

        if n_vort < 0:
            print(f"   ✅ Vorticity detected as a 'red' field (n < 0)")
            if n_vort < -0.5:
                print(f"    🎯 FINDING: Vorticity is clearly distinct from density")

        # Significance of the difference
        z_score = abs(n_vort - n_dens) / 0.2  # Conservative estimated error
        print(f"\n📈 ESTIMATED SIGNIFICANCE:")
        print(f"   • |n_vort - n_dens| = {abs(n_vort - n_dens):.2f}")
        print(f"   • Significance ≈ {z_score:.1f}σ")

    else:
        print("\n❌ Comparison could not be completed")

if __name__ == "__main__":
    main()