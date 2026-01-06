#!/usr/bin/env python3
"""
IMPROVED TOMOGRAPHIC ANALYSIS - 2 WIDE BINS
"""                                                               
import numpy as np          
import healpy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import json
import warnings
warnings.filterwarnings('ignore')

# Configuration
NSIDE = 64
LMAX = 100  # Reduced for greater stability
BIN_SIZE = 15  # Wider to reduce variance
np.random.seed(42)

def process_robust_redshift_bin(z_min, z_max, data, bin_name):
    """Robust version handling low statistics"""
    print(f"\n📊 PROCESSING: {z_min} < z ≤ {z_max} ({bin_name})")

    # Redshift filter
    mask = (data['Z'] > z_min) & (data['Z'] <= z_max)
    data_bin = data[mask]

    if len(data_bin) < 50000:
        print(f"   ❌ Sample too small: {len(data_bin)} galaxies")
        return None

    ra = np.array(data_bin['RA'])
    dec = np.array(data_bin['DEC'])
    vdisp = np.array(data_bin['VDISP'])

    # Verify uniform distribution
    npix = hp.nside2npix(NSIDE)
    theta = np.radians(90.0 - dec)
    phi = np.radians(ra)
    pix = hp.ang2pix(NSIDE, theta, phi)
    populated_pixels = len(np.unique(pix))
    f_sky_raw = populated_pixels / npix

    if f_sky_raw < 0.15:
        print(f"   ⚠️ f_sky too low: {f_sky_raw:.3f}")
        return None

    # Split with balance verification
    n_total = len(ra)
    for attempt in range(3):
        indices = np.random.permutation(n_total)
        cut = n_total // 2

        # Verify that both splits cover similar areas
        pix_A = hp.ang2pix(NSIDE,
                          np.radians(90.0 - dec[indices[:cut]]),
                          np.radians(ra[indices[:cut]]))
        pix_B = hp.ang2pix(NSIDE,
                          np.radians(90.0 - dec[indices[cut:]]),
                          np.radians(ra[indices[cut:]]))

        if (len(np.unique(pix_A)) > 0.8 * populated_pixels and
            len(np.unique(pix_B)) > 0.8 * populated_pixels):
            break

    # Generate maps
    def generate_map(idx):
        npix = hp.nside2npix(NSIDE)
        theta = np.radians(90.0 - dec[idx])
        phi = np.radians(ra[idx])
        pix_local = hp.ang2pix(NSIDE, theta, phi)

        counts = np.bincount(pix_local, minlength=npix)
        vdisp_sum = np.bincount(pix_local, weights=vdisp[idx], minlength=npix)

        vdisp_map = np.zeros(npix)
        mask_good = counts > 0
        if np.sum(mask_good) == 0:
            return np.zeros(npix), 0.0

        vdisp_map[mask_good] = vdisp_sum[mask_good] / counts[mask_good]
        global_mean = np.mean(vdisp[idx])

        delta_map = np.zeros(npix)
        delta_map[mask_good] = (vdisp_map[mask_good] - global_mean) / global_mean

        return delta_map, np.sum(mask_good) / npix

    map_A, fsky_A = generate_map(indices[:cut])
    map_B, fsky_B = generate_map(indices[cut:])

    if fsky_A < 0.1 or fsky_B < 0.1:
        print(f"   ❌ Insufficient f_sky: {fsky_A:.3f}, {fsky_B:.3f}")
        return None

    fsky_avg = (fsky_A + fsky_B) / 2
    print(f"   • Galaxies: {n_total:,}")
    print(f"   • f_sky: {fsky_avg:.3f}")
    print(f"   • Populated pixels: {populated_pixels:,}/{npix:,}")

    # Cross-spectrum with verification
    cl_raw = hp.anafast(map_A, map_B, lmax=LMAX)
    ell = np.arange(len(cl_raw))

    # Smooth spectrum to reduce noise
    cl_smooth = np.zeros_like(cl_raw)
    for l in range(len(cl_raw)):
        l_min = max(0, l-2)
        l_max = min(len(cl_raw)-1, l+2)
        cl_smooth[l] = np.mean(cl_raw[l_min:l_max+1])

    cl_corrected = cl_smooth / fsky_avg

    # More permissive binning
    ell_binned, cl_binned, err_binned = [], [], []

    for i in range(5, LMAX, BIN_SIZE):
        l_low = i
        l_high = min(i + BIN_SIZE, LMAX)
        mask_bin = (ell >= l_low) & (ell < l_high)

        if np.sum(mask_bin) > 2:
            cl_values = cl_corrected[mask_bin]
            cl_mean = np.mean(cl_values)
            cl_std = np.std(cl_values)

            # Accept bin if it's not clearly noise
            if cl_mean > -3*cl_std:  # Allow slightly negative values
                ell_binned.append(np.mean(ell[mask_bin]))
                cl_binned.append(max(cl_mean, 1e-12))  # Avoid zero
                err_binned.append(cl_std / np.sqrt(len(cl_values)))

    if len(ell_binned) < 4:
        print(f"   ❌ Only {len(ell_binned)} valid bins")
        return None

    print(f"   • Valid bins: {len(ell_binned)}")
    print(f"   • C_ℓ Range: {min(cl_binned):.2e} to {max(cl_binned):.2e}")

    # Robust Fit
    try:
        ell_arr = np.array(ell_binned)
        cl_arr = np.array(cl_binned)
        err_arr = np.array(err_binned)

        # Use absolute value log to avoid issues with negatives
        log_ell = np.log(ell_arr)
        log_cl = np.log(np.abs(cl_arr))

        # Simple fit with polyfit (more robust than curve_fit)
        coeffs = np.polyfit(log_ell, log_cl, 1, w=1.0/(err_arr/cl_arr + 1e-10))
        n_omega = coeffs[0]

        # Error by simple bootstrap
        n_bootstrap = []
        for _ in range(100):
            idx = np.random.choice(len(ell_arr), size=len(ell_arr), replace=True)
            if len(np.unique(ell_arr[idx])) > 3:
                coeffs_bs = np.polyfit(log_ell[idx], log_cl[idx], 1)
                n_bootstrap.append(coeffs_bs[0])

        n_error = np.std(n_bootstrap) if n_bootstrap else 0.5

        # R²
        cl_pred = np.exp(n_omega * log_ell + coeffs[1])
        ss_res = np.sum((cl_arr - cl_pred)**2)
        ss_tot = np.sum((cl_arr - np.mean(cl_arr))**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        print(f"   ✅ n_ω = {n_omega:.3f} ± {n_error:.3f}")
        print(f"   • R² = {r2:.3f}")

        return {
            'ell': ell_arr.tolist(),
            'cl': cl_arr.tolist(),
            'err': err_arr.tolist(),
            'n_omega': float(n_omega),
            'n_error': float(n_error),
            'r2': float(r2),
            'n_galaxies': n_total,
            'f_sky': float(fsky_avg)
        }

    except Exception as e:
        print(f"   ❌ Fit error: {e}")
        return None

def main():
    print("=" * 60)
    print("📡 IMPROVED TOMOGRAPHIC ANALYSIS - 2 WIDE BINS")
    print("=" * 60)

    # Load data
    print("\n📥 LOADING DATA...")
    try:
        data = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
        print(f"   • Total galaxies: {len(data):,}")

        # Filters
        mask = ((data['VDISP'] >= 150) & (data['VDISP'] <= 500) &
                (data['Z'] >= 0.4) & (data['Z'] <= 1.0))
        data = data[mask]
        print(f"   • After filters: {len(data):,}")

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # DEFINE 2 WIDE BINS (not 4)
    redshift_bins = [
        (0.4, 0.7, "LOW Bin: 0.4 < z ≤ 0.7"),
        (0.7, 1.0, "HIGH Bin: 0.7 < z ≤ 1.0")
    ]

    # Process
    results = {}
    for z_min, z_max, label in redshift_bins:
        key = f"{z_min}-{z_max}"
        res = process_robust_redshift_bin(z_min, z_max, data, label)
        results[key] = res

    # Create plot if there's at least one result
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['red', 'blue']

    plot_data = []

    for i, (z_range, res) in enumerate(results.items()):
        if res is not None:
            ell = np.array(res['ell'])
            cl = np.array(res['cl'])
            err = np.array(res['err'])
            n = res['n_omega']
            n_err = res['n_error']
            r2 = res['r2']

            # Plot
            ax.errorbar(ell, cl, yerr=err, fmt='o',
                        color=colors[i % len(colors)],
                        alpha=0.7, capsize=4,
                        label=f'{z_range}: n={n:.2f}±{n_err:.2f} (R²={r2:.2f})')

            # Power law
            ell_fine = np.linspace(ell.min(), ell.max(), 100)
            A = np.mean(np.log(cl) - n * np.log(ell))
            cl_fit = np.exp(n * np.log(ell_fine) + A)
            ax.plot(ell_fine, cl_fit, '--',
                    color=colors[i % len(colors)], alpha=0.5)

            plot_data.append((z_range, res))

    if plot_data:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Multipole ℓ', fontsize=14)
        ax.set_ylabel('C_ℓ (Cross-Spectrum)', fontsize=14)
        ax.set_title('Vorticity Spectrum Evolution (2 wide bins)', fontsize=16)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
        plt.tight_layout()
        plt.savefig('vorticity_2bins.png', dpi=150, bbox_inches='tight')
        print(f"\n📈 Plot saved: vorticity_2bins.png")
        plt.close()
    else:
        print("\n❌ Insufficient data for plotting")

    # Save results
    output_data = {}
    for key, res in results.items():
        if res is not None:
            output_data[key] = res

    with open('results_2bins.json', 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n" + "=" * 60)
    print("✅ ANALYSIS COMPLETED")
    print("=" * 60)

    # Summary
    if output_data:
        print("\n📋 RESULTS:")
        for key, stats in output_data.items():
            print(f"   • {key}: n_ω = {stats['n_omega']:.3f} ± {stats['n_error']:.3f}")
            print(f"     Galaxies: {stats['n_galaxias']:,}, f_sky: {stats['f_sky']:.3f}")

        # Comparison with global result
        print(f"\n💡 COMPARISON WITH GLOBAL RESULT:")
        print(f"   • Global: n_ω = -1.232 ± 0.121 (from your main analysis)")

        # Evolution
        if len(output_data) > 1:
            keys = list(output_data.keys())
            n_low = output_data[keys[0]]['n_omega']
            n_high = output_data[keys[1]]['n_omega']
            print(f"\n🔍 TEMPORAL TREND:")
            print(f"   • n_ω(low z) - n_ω(high z) = {n_low - n_high:.3f}")
            if n_low < n_high:  # More negative at low z
                print(f"   → Vorticity becomes more 'red' (structured) over time")
    else:
        print("\n❌ No valid results obtained")

if __name__ == "__main__":
    main()