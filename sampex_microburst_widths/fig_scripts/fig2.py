"""
This script generates a histogram of all microburst widths on the left panel, and
the widths as a function of AE in the right panel.

Parameters
----------
catalog_name: str
    The name of the catalog in the config.PROJECT_DIR/data/ directory.
r2_thresh: float
    The adjusted R^2 threshold for the fits. I chose a default value of 0.9.
max_width: float
    Maximum microburst width (FWHM) in seconds to histogram. A good default is
    0.25 [seconds]
"""
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

plt.rcParams.update({'font.size': 13})

### Script parameters ###
catalog_name = 'microburst_catalog_02.csv'
r2_thresh = 0.9
max_width_ms = 500
width_bins = np.linspace(0, max_width_ms+0.001, num=50)
ae_bins = [0, 100, 300]
width_key = 'fwhm_ms'

# Load the catalog, drop the NaN values, and filter by the max_width and
# R^2 values.
df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df.dropna(inplace=True)
df['width_ms'] = 1000*df['width_s'] # Convert seconds to ms.
df['fwhm_ms'] = 1000*df['fwhm']
df = df[df['width_ms'] < max_width_ms]
df['fwhm_ms'] = df['fwhm_ms'].abs()
df = df[df.adj_r2 > r2_thresh]

quantiles = [.25, .50, .75]
width_percentiles = df[width_key].quantile(q=quantiles)

fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

# Left panel histogram and statistics.
ax[0].hist(df[width_key], bins=width_bins, color='k', histtype='step', density=True)
s = (
    f"Percentiles [ms]"
    f"\n25%: {(width_percentiles.loc[0.25]).round().astype(int)}"
    f"\n50%: {(width_percentiles.loc[0.50]).round().astype(int)}"
    f"\n75%: {(width_percentiles.loc[0.75]).round().astype(int)}"
)
ax[0].text(0.64, 0.9, s, 
        ha='left', va='top', transform=ax[0].transAxes
        )
plt.suptitle('Distribution of > 1 MeV Microburst Duration\nSAMPEX/HILT')
# Left panel tweaks
ax[0].set_xlim(0, max_width_ms)
ax[0].set_ylabel('Probability Density')
ax[0].set_xlabel('FWHM [ms]')

# Right panel histogram and statistics for the first two categories.
for start_ae, end_ae in zip(ae_bins[:-1], ae_bins[1:]):
    df_flt = df[(df['AE'] > start_ae) & (df['AE'] < end_ae)]

    ax[1].hist(df_flt[width_key], bins=width_bins, histtype='step', density=True,
            label=f'{start_ae} < AE [nT] < {end_ae}', lw=2)
    print(f'Median microburst width for {start_ae} < AE [nT] < {end_ae} is '
          f'{round(df_flt["fwhm"].median(), 2)} s | N = {df_flt.shape[0]}')

# Last category that is AE > ae_bins[-1]
df_flt = df[df['AE'] > ae_bins[-1]]
ax[1].hist(df_flt[width_key], bins=width_bins, histtype='step', density=True,
    label=f'AE [nT] > {ae_bins[-1]}', lw=2)
print(f'Median microburst width for {start_ae} < AE [nT] < {end_ae} is '
          f'{round(df_flt[width_key].median(), 2)} s | N = {df_flt.shape[0]}')

ax[1].legend(loc='center right', fontsize=12)
ax[1].set_xlabel('FWHM [ms]')

# Subplot labels
# for a, label in zip(ax, string.ascii_lowercase):
ax[0].text(0, 0.99, f'(a) All Microbursts', va='top', transform=ax[0].transAxes, 
    weight='bold', fontsize=15)
ax[1].text(0, 0.99, f'(b) As a function of AE', va='top', transform=ax[1].transAxes, 
    weight='bold', fontsize=15)

plt.tight_layout()
plt.show()