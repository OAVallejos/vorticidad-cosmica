#!/usr/bin/env python3
"""
SHUFFLE TEST: Demonstrate that the signal disappears when VDISP is randomized
"""                                                     
import numpy as np
import healpy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
NSIDE = 64
LMAX = 150
BIN_SIZE = 10
np.random.seed(42)

def analyze_shuffle(data, n_shuffles=10):
    """
    Analyzes multiple realizations with randomized VDISP
    """
    print(f"\n🎲 SHUFFLE TEST: {n_shuffles} realizations")

    # Filter (same as before)
    mask = ((data['Z'] >= 0.4) & (data['Z'] <= 0.7) &
            (data['VDISP'] >= 150) & (data['VDISP'] <= 500))
    data_sub = data[mask]

    ra = np.array(data_sub['RA'])
    dec = np.array(data_sub['DEC'])
    vdisp_original = np.array(data_sub['VDISP'])

    n_total = len(ra)
    shuffle_results = []

    for shuffle_idx in range(n_shuffles):
        print(f"\n   Shuffle {shuffle_idx+1}/{n_shuffles}:", end=" ")

        # CRITICAL SHUFFLE: Randomize VDISP while maintaining positions
        vdisp_shuffled = np.random.permutation(vdisp_original)

        # Split-half
        indices = np.random.permutation(n_total)
        cut = n_total // 2

        # Map generation function
        def generate_map(idx, vdisp_vals):
            npix = hp.nside2npix(NSIDE)
            theta = np.radians(90.0 - dec[idx])
            phi = np.radians(ra[idx])
            pix = hp.ang2pix(NSIDE, theta, phi)

            counts = np.bincount(pix, minlength=npix)
            v_sum = np.bincount(pix, weights=vdisp_vals[idx], minlength=npix)

            vdisp_map = np.zeros(npix)
            mask_good = counts > 0
            vdisp_map[mask_good] = v_sum[mask_good] / counts[mask_good]

            global_mean = np.mean(vdisp_vals[idx])
            delta_map = np.zeros(npix)
            delta_map[mask_good] = (vdisp_map[mask_good] - global_mean) / global_mean

            f_sky = np.sum(mask_good) / npix
            return delta_map, f_sky

        # Maps with randomized VDISP
        map_A, fsky_A = generate_map(indices[:cut], vdisp_shuffled)
        map_B, fsky_B = generate_map(indices[cut:], vdisp_shuffled)
        fsky_avg = (fsky_A + fsky_B) / 2

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
                if cl_mean > 0:
                    ell_binned.append(np.mean(ell[mask_bin]))
                    cl_binned.append(cl_mean)

        if len(ell_binned) < 4:
            print("Insufficient bins", end="")
            continue

        # Calculate n
        log_ell = np.log(ell_binned)
        log_cl = np.log(cl_binned)

        try:
            coeffs = np.polyfit(log_ell, log_cl, 1)
            n = coeffs[0]
            shuffle_results.append(n)
            print(f"n = {n:.3f}", end="")
        except:
            print("Fit error", end="")

    return shuffle_results

def analyze_original(data):
    """
    Analyze ORIGINAL data (no shuffle) for comparison
    """
    print("\n📊 ANALYZING ORIGINAL DATA (no shuffle)")

    mask = ((data['Z'] >= 0.4) & (data['Z'] <= 0.7) &
            (data['VDISP'] >= 150) & (data['VDISP'] <= 500))
    data_sub = data[mask]

    ra = np.array(data_sub['RA'])
    dec = np.array(data_sub['DEC'])
    vdisp = np.array(data_sub['VDISP'])

    n_total = len(ra)
    indices = np.random.permutation(n_total)
    cut = n_total // 2

    def generate_map(idx):
        npix = hp.nside2npix(NSIDE)
        theta = np.radians(90.0 - dec[idx])
        phi = np.radians(ra[idx])
        pix = hp.ang2pix(NSIDE, theta, phi)

        counts = np.bincount(pix, minlength=npix)
        v_sum = np.bincount(pix, weights=vdisp[idx], minlength=npix)

        vdisp_map = np.zeros(npix)
        mask_good = counts > 0
        vdisp_map[mask_good] = v_sum[mask_good] / counts[mask_good]

        global_mean = np.mean(vdisp[idx])
        delta_map = np.zeros(npix)
        delta_map[mask_good] = (vdisp_map[mask_good] - global_mean) / global_mean

        f_sky = np.sum(mask_good) / npix
        return delta_map, f_sky

    map_A, fsky_A = generate_map(indices[:cut])
    map_B, fsky_B = generate_map(indices[cut:])
    fsky_avg = (fsky_A + fsky_B) / 2

    cl_raw = hp.anafast(map_A, map_B, lmax=LMAX)
    ell = np.arange(len(cl_raw))
    cl_corrected = cl_raw / fsky_avg

    # Binning and fitting
    ell_binned, cl_binned = [], []
    for i in range(2, LMAX, BIN_SIZE):
        mask_bin = (ell >= i) & (ell < min(i + BIN_SIZE, LMAX))
        if np.sum(mask_bin) > 0:
            cl_mean = np.mean(cl_corrected[mask_bin])
            if cl_mean > 0:
                ell_binned.append(np.mean(ell[mask_bin]))
                cl_binned.append(cl_mean)

    log_ell = np.log(ell_binned)
    log_cl = np.log(cl_binned)
    coeffs, cov = np.polyfit(log_ell, log_cl, 1, cov=True)
    n_original = coeffs[0]
    n_err = np.sqrt(cov[0,0])

    return n_original, n_err, np.array(ell_binned), np.array(cl_binned)

def main():
    print("=" * 60)
    print("🔬 DEFINITIVE TEST: VDISP SHUFFLE")
    print("=" * 60)

    # Load data
    print("\n📥 LOADING DATA...")
    try:
        data = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
        print(f"   • Total galaxies: {len(data):,}")
    except Exception as e:
        print(f"❌ Error: {e}")
        return

    # 1. Analyze ORIGINAL data
    n_original, n_err, ell_orig, cl_orig = analyze_original(data)
    print(f"   • n_original = {n_original:.3f} ± {n_err:.3f}")

    # 2. Analyze multiple SHUFFLES
    n_shuffles = 500  # Number of random realizations
    shuffle_results = analyze_shuffle(data, n_shuffles)

    if not shuffle_results:
        print("\n❌ Shuffles could not be completed")
        return

    # 3. Shuffle statistics
    n_shuffle_arr = np.array(shuffle_results)
    print(f"\n\n📊 STATISTICS FOR {len(n_shuffle_arr)} SHUFFLES:")
    print(f"   • Mean n_shuffle: {np.mean(n_shuffle_arr):.3f}")
    print(f"   • Std n_shuffle: {np.std(n_shuffle_arr):.3f}")
    print(f"   • Range: [{np.min(n_shuffle_arr):.3f}, {np.max(n_shuffle_arr):.3f}]")

    # 4. Significance
    z_score = abs(n_original - np.mean(n_shuffle_arr)) / np.std(n_shuffle_arr)
    print(f"\n🎯 STATISTICAL SIGNIFICANCE:")
    print(f"   • n_original - <n_shuffle> = {n_original - np.mean(n_shuffle_arr):.3f}")
    print(f"   • z-score = {z_score:.2f}σ")

    # 5. Comparative Histogram
    plt.figure(figsize=(10, 6))

    # Shuffle histogram
    plt.hist(n_shuffle_arr, bins=10, alpha=0.7, color='gray',
             label=f'Shuffles (n={len(n_shuffle_arr)})', density=True)

    # Line for original data
    plt.axvline(x=n_original, color='red', linewidth=3,
                label=f'Original: n={n_original:.3f}')

    # Line for shuffle mean
    plt.axvline(x=np.mean(n_shuffle_arr), color='blue', linestyle='--',
                label=f'Shuffle mean: {np.mean(n_shuffle_arr):.3f}')

    plt.xlabel('Spectral Index n', fontsize=14)
    plt.ylabel('Probability Density', fontsize=14)
    plt.title('Shuffle Test: Randomized VDISP n vs Original', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add text with statistics
    textstr = '\n'.join([
        f'Original: n = {n_original:.3f}',
        f'Shuffle mean: {np.mean(n_shuffle_arr):.3f}',
        f'Shuffle std: {np.std(n_shuffle_arr):.3f}',
        f'z-score: {z_score:.2f}σ'
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', bbox=props)

    plt.tight_layout()
    plt.savefig('vortical_shuffle_test.png', dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved: vortical_shuffle_test.png")
    plt.close()

    # 6. Interpretation
    print("\n" + "=" * 60)
    print("💡 INTERPRETATION OF THE SHUFFLE TEST")
    print("=" * 60)

    if z_score > 3.0:
        print("✅ **CONCLUSIVE RESULT**:")
        print(f"   The vorticity signal (n = {n_original:.3f}) is")
        print(f"   statistically distinct from random noise ({z_score:.1f}σ)")
        print("   → The spatial correlation of VDISP is PHYSICAL, not statistical")
    elif z_score > 1.5:
        print("⚠️ **MODERATE EVIDENCE**:")
        print(f"   There are indications of a real signal ({z_score:.1f}σ)")
        print("   → More data or additional analysis is required")
    else:
        print("❓ **INCONCLUSIVE RESULT**:")
        print(f"   Cannot rule out statistical fluctuation ({z_score:.1f}σ)")
        print("   → Review methodology or increase sample size")

    # Comparison with previous results
    print(f"\n🔍 CONTEXT FROM PREVIOUS RESULTS:")
    print(f"   • Global (0.4<z<1.0): n_ω = -1.232 ± 0.121")
    print(f"   • Low bin (0.4<z<0.7): n_ω = -1.675 ± 0.081")
    print(f"   • Density control: n = -2.295 ± 0.081")
    print(f"   • This shuffle test: mean = {np.mean(n_shuffle_arr):.3f}")

if __name__ == "__main__":
    main()