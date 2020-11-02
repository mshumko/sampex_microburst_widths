"""
This script makes Figure 4: a histogram of the microburst durations as a 
function of AE.
"""
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

### Script parameters ###
catalog_name = 'microburst_catalog_02.csv'
max_width=0.5
width_bins=np.linspace(0, max_width, num=20)
ae_bins = [0, 100, 300, 1000]
r2_thresh = 0.9

df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df.dropna(inplace=True)

df['fwhm'] = df['fwhm'].abs()
# Filter by the R^2 value with the microburst FWHM 
df = df[(df['adj_r2'] > r2_thresh) & (df['fwhm'] < max_width)]

_, ax = plt.subplots(1, len(ae_bins)-1, figsize=(12, 5), sharey=True, sharex=True)

for ax_i, start_ae, end_ae, label_i in zip(ax, ae_bins[:-1], ae_bins[1:], string.ascii_lowercase):
    df_flt = df[(df['AE'] > start_ae) & (df['AE'] < end_ae)]
    ax_i.hist(df_flt['fwhm'], bins=width_bins, histtype='step', density=True,
            label=f'{start_ae} < AE < {end_ae}', color='k', lw=1)
    ax_i.text(0, 0.99, f'({label_i}) {start_ae} < AE [nT] < {end_ae}', va='top', 
            transform=ax_i.transAxes, fontsize=15)
    ax_i.set_xlabel('Microburst FWHM [s]')

ax[0].set_ylabel('Probability density')
plt.suptitle('Distribution of SAMPEX microburst duration as a function of AE', fontsize=20)
plt.tight_layout()
plt.show()