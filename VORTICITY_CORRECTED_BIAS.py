import numpy as np
import healpy as hp
from astropy.table import Table
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Configuration
NSIDE = 64
LMAX = 120
N_SHUFFLES = 500

def analyze_pure_vorticity():
    print("=" * 60)
    print("🧹 PURE VORTICITY: Geometric Bias Correction")
    print("=" * 60)

    # 1. Load data
    data = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
    mask = (data['Z'] >= 0.4) & (data['Z'] <= 0.7) & (data['VDISP'] >= 150)
    data_sub = data[mask]

    ra = np.array(data_sub['RA'])
    dec = np.array(data_sub['DEC'])
    vdisp = np.array(data_sub['VDISP'])
    n_total = len(ra)

    def generate_vort_map(v_vals, indices_part):
        npix = hp.nside2npix(NSIDE)
        ra_p, dec_p, v_p = ra[indices_part], dec[indices_part], v_vals[indices_part]

        theta, phi = np.radians(90.0 - dec_p), np.radians(ra_p)
        pix = hp.ang2pix(NSIDE, theta, phi)

        counts = np.bincount(pix, minlength=npix)
        v_sum = np.bincount(pix, weights=v_p, minlength=npix)

        m = np.zeros(npix)
        mask_pix = counts > 0
        # Calculate normalized fluctuation map
        m[mask_pix] = (v_sum[mask_pix]/counts[mask_pix] - np.mean(v_p)) / np.mean(v_p)
        return m, np.sum(mask_pix)/npix

    # 2. Real Spectrum
    print(f"📡 Processing {n_total:,} galaxies...")
    idx = np.random.permutation(n_total)
    cut = n_total // 2
    mA, fA = generate_vort_map(vdisp, idx[:cut])
    mB, fB = generate_vort_map(vdisp, idx[cut:])
    cl_real = hp.anafast(mA, mB, lmax=LMAX) / ((fA+fB)/2)

    # 3. Bias Spectrum (Shuffles)
    print(f"🎲 Calculating bias ({N_SHUFFLES} shuffles)...")
    cl_shuf_list = []
    for i in range(N_SHUFFLES):
        v_shuf = np.random.permutation(vdisp)
        mA_s, fA_s = generate_vort_map(v_shuf, idx[:cut])
        mB_s, fB_s = generate_vort_map(v_shuf, idx[cut:])
        cl_shuf_list.append(hp.anafast(mA_s, mB_s, lmax=LMAX) / ((fA_s+fB_s)/2))

    cl_bias = np.mean(cl_shuf_list, axis=0)
    cl_pure = cl_real - cl_bias

    # 4. n_omega Fit
    ell = np.arange(len(cl_pure))
    mask_fit = (ell > 10) & (ell < 100) & (cl_pure > 0)
    n_pure = np.polyfit(np.log(ell[mask_fit]), np.log(cl_pure[mask_fit]), 1)[0]

    print(f"\n🎯 CLEAN FINAL RESULT:")
    print(f"   • n_omega (corrected) = {n_pure:.3f}")

    # Plot
    plt.figure(figsize=(10,6))
    plt.loglog(ell[ell>0], cl_real[ell>0], label='Real (Signal + Bias)', alpha=0.5)
    plt.loglog(ell[ell>0], cl_bias[ell>0], '--', label='Geometric Bias', color='red')
    plt.loglog(ell[mask_fit], cl_pure[mask_fit], 'ko', label=f'Pure (Corrected): n={n_pure:.3f}')
    plt.title("DESI Vorticity Spectrum: Bias Correction")
    plt.xlabel("Multipole ℓ"); plt.ylabel("C_ℓ"); plt.legend(); plt.grid(True, alpha=0.2)
    plt.savefig('vorticity_final_corrected.png')
    print("📈 Plot saved: vorticity_final_corrected.png")

if __name__ == "__main__":
    analyze_pure_vorticity()