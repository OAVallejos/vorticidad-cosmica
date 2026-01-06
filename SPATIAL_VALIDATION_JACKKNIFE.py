#!/usr/bin/env python3
"""
SPATIAL VALIDATION (JACKKNIFE) - Isotropy of n_ω
Divides the sky into 4 regions to confirm the universality of the signal.
"""

import numpy as np
import healpy as hp
from astropy.table import Table
import matplotlib.pyplot as plt

# Configuration
NSIDE = 64
LMAX = 100

def run_region_analysis(data_reg, label):
    """Calculates n_omega for a spatial subset"""
    ra = np.array(data_reg['RA'])
    dec = np.array(data_reg['DEC'])
    vdisp = np.array(data_reg['VDISP'])

    # Split-half to clean noise
    idx = np.random.permutation(len(ra))
    cut = len(ra) // 2

    def get_cl(indices):
        npix = hp.nside2npix(NSIDE)
        # Convert RA/DEC to Healpix pixels
        pix = hp.ang2pix(NSIDE, np.radians(90-dec[indices]), np.radians(ra[indices]))
        counts = np.bincount(pix, minlength=npix)
        v_sum = np.bincount(pix, weights=vdisp[indices], minlength=npix)
        
        m = np.zeros(npix)
        g_m = np.mean(vdisp[indices])
        # Calculate overdensity/fluctuation map
        m[counts>0] = (v_sum[counts>0]/counts[counts>0] - g_m) / g_m
        fsky = np.sum(counts>0)/npix
        return m, fsky

    m1, f1 = get_cl(idx[:cut])
    m2, f2 = get_cl(idx[cut:])
    
    # Calculate cross-power spectrum corrected by sky fraction
    cl = hp.anafast(m1, m2, lmax=LMAX) / ((f1+f2)/2)
    ell = np.arange(len(cl))[5:LMAX] # Cut low/noisy multipoles
    cl = cl[5:LMAX]

    # Log-log linear fit
    mask = cl > 0
    coeffs = np.polyfit(np.log(ell[mask]), np.log(cl[mask]), 1)
    return coeffs[0]

# 1. Load data
data = Table.read('DATASET_LRG_VDISP_FLUXR_FINAL.fits')
data = data[(data['Z'] >= 0.4) & (data['Z'] <= 0.7)] # Highest SNR bin

# 2. Define Quadrants
# Mean RA ~ 180, Mean DEC ~ 30
regions = [
    ("NE", (data['RA'] > 180) & (data['DEC'] > 30)),
    ("NW", (data['RA'] <= 180) & (data['DEC'] > 30)),
    ("SE", (data['RA'] > 180) & (data['DEC'] <= 30)),
    ("SW", (data['RA'] <= 180) & (data['DEC'] <= 30))
]

print(f"{'Region':<10} | {'Galaxies':<12} | {'n_ω':<10}")
print("-" * 40)

results_n = []
for name, mask in regions:
    sub_data = data[mask]
    if len(sub_data) > 10000:
        n = run_region_analysis(sub_data, name)
        results_n.append(n)
        print(f"{name:<10} | {len(sub_data):<12,} | {n:<10.3f}")

# 3. Jackknife Statistics
n_mean = np.mean(results_n)
n_std = np.std(results_n)
print("-" * 40)
print(f"JACKKNIFE AVERAGE: n_ω = {n_mean:.3f} ± {n_std:.3f}")