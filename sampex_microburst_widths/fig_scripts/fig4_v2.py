"""
This script generates Figure 4: a plot of the marginalized microburst duration 
as a function of L and MLT. This script is similar to fig4.py except that
I add two panels showing the FWHM-L distribution for 2 distinct MLT regions.

Parameters
----------
catalog_name: str
    The name of the catalog in the config.PROJECT_DIR/data/ directory.
r2_thresh: float
    The adjusted R^2 threshold for the fits. I chose a default value of 0.9.
max_width_ms: float
    Maximum microburst width (FWHM) in milliseconds to histogram. A good default is
    250 ms.
mlt_regions: np.array
    A 2x2 array with the rows representing the start and end MLT.
"""
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from sampex_microburst_widths import config

plt.rcParams.update({'font.size': 15})

### Script parameters ###
catalog_name = 'microburst_catalog_04.csv'
r2_thresh = 0.9
max_width_ms = 250
width_bins = np.linspace(0, max_width_ms+0.001, num=50)
L_bins = np.linspace(2, 8.1, num=50)
MLT_bins = np.linspace(0, 24, num=50)
mlt_regions = np.array([[22, 2], [3, 6]])  # For the L-shell distributions

# Load the catalog, drop the NaN values, and filter by the max_width and
# R^2 values.
df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df.dropna(inplace=True)
df['width_ms'] = 1000*df['width_s'] # Convert seconds to ms.
df['fwhm_ms'] = 1000*df['fwhm']
initial_shape = df.shape[0]
df = df[df['width_ms'] < max_width_ms]
df['fwhm_ms'] = df['fwhm_ms'].abs()
df = df[df.adj_r2 > r2_thresh]
print(f'The {initial_shape} microbursts were filtered down to {df.shape[0]} microbursts.')

# Create a histogram of L-FWHM and MLT-FWHM for all microbursts
H_L, _, _ = np.histogram2d(df['L_Shell'], df['fwhm_ms'],
                        bins=[L_bins, width_bins])
H_MLT, _, _ = np.histogram2d(df['MLT'], df['fwhm_ms'],
                        bins=[MLT_bins, width_bins])

# Create a histogram of L-FWHM for the two mlt_regions
H_L_regions = np.nan*np.zeros((2, *H_L.shape))
for i, mlt_region in enumerate(mlt_regions):
    if mlt_region[0] > mlt_region[1]:  # Presumably the region crosses over 0 MLT.
        df_flt = df[(df['MLT'] > mlt_region[0]) | (df['MLT'] < mlt_region[1])]
    else:
        df_flt = df[(df['MLT'] > mlt_region[0]) & (df['MLT'] < mlt_region[1])]
    H_L_regions[i, :, :], _, _ = np.histogram2d(df_flt['L_Shell'], df_flt['fwhm_ms'],
                                bins=[L_bins, width_bins])


# Make the FWHM-L and FWHM-MLT plots.
_, ax = plt.subplots(2, 2, figsize=(8, 6))

p_L = ax[0, 0].pcolormesh(L_bins, width_bins, H_L.T, vmin=0)
# plt.colorbar(p_L, ax=ax[0, 0], orientation='horizontal', label='Number of microbursts')
ax[0, 0].set_xlabel('L-shell')
ax[0, 0].set_ylabel('FWHM [ms]')

p_MLT = ax[0, 1].pcolormesh(MLT_bins, width_bins, H_MLT.T, vmin=0)
# plt.colorbar(p_MLT, ax=ax[0, 1], orientation='horizontal', label='Number of microbursts')
ax[0, 1].set_xlabel('MLT')
ax[0, 1].set_ylabel('FWHM [ms]')

# Make the FWHM-L plot for the two MLT regions.
for i, H_L_region in enumerate(H_L_regions):
    p_L = ax[1, i].pcolormesh(L_bins, width_bins, H_L_region.T, vmin=0, vmax=20)
    # plt.colorbar(p_L, ax=ax[1, i], orientation='horizontal', label='Number of microbursts')
    ax[1, i].set_xlabel('L-shell')
    ax[1, i].set_ylabel('FWHM [ms]')

subplot_text = [
    f'(a) All microbursts', 
    f'(b)',
    f'(c) {mlt_regions[0, 0]} < MLT < {mlt_regions[0, 1]}', 
    f'(d) {mlt_regions[1, 0]} < MLT < {mlt_regions[1, 1]}'
                ]
i=0
for ax_row in ax:
    for ax_i in ax_row:
        # annotate_str = f'({string.ascii_lowercase[i]})'
        ax_i.text(0, 1, subplot_text[i], va='top', color='white', weight='bold', 
                    transform=ax_i.transAxes, fontsize=20)
        i+=1

plt.suptitle(f'Distribution of SAMPEX microburst durations in L and MLT', fontsize=20)

plt.tight_layout()
plt.show()