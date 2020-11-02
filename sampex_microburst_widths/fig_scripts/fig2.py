"""
This script makes Figure 2: a dial plot of the microburst width as a function of L and MLT
"""
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from sampex_microburst_widths import config
from sampex_microburst_widths.stats import dial_plot

plt.rcParams.update({'font.size': 13})

catalog_name = 'microburst_catalog_02.csv'

### Script parameters
statistics_thresh=100 # Don't calculate stats if less microbursts in the bin.
percentiles = np.array([25, 50, 75])
r2_thresh = 0.9
max_width = 0.5
L_bins = np.linspace(2, 8.1, num=20)
L_labels = [2,4,6]
MLT_bins = np.linspace(0, 24, num=40)

df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df.dropna(inplace=True)
df = df[df['width_s'] < max_width]
df['fwhm'] = df['fwhm'].abs()
df = df[df.adj_r2 > r2_thresh]

num_microbursts_H, _, _ = np.histogram2d(df['MLT'], df['L_Shell'],
                                        bins=[MLT_bins, L_bins])
H = np.nan*np.zeros(
    (len(MLT_bins), len(L_bins), len(percentiles))
                    )

for i, (start_MLT, end_MLT) in enumerate(zip(MLT_bins[:-1], MLT_bins[1:])):
    for j, (start_L, end_L) in enumerate(zip(L_bins[:-1], L_bins[1:])):
        df_flt = df.loc[(
            (df['MLT'] > start_MLT) &  (df['MLT'] < end_MLT) &
            (df['L_Shell'] > start_L) &  (df['L_Shell'] < end_L)
            ), 'fwhm']
        if df_flt.shape[0] >= statistics_thresh:
            H[i, j, :] = df_flt.quantile(percentiles/100)

fig = plt.figure(figsize=(10, 8))
ax = [plt.subplot(2, 2, i, projection='polar') for i in range(1, 5)]

for i, ax_i in enumerate(ax[:-1]):
    d = dial_plot.Dial(ax_i, MLT_bins, L_bins, H[:, :, i])
    d.draw_dial(L_labels=L_labels,
            colorbar_kwargs={'label':f'microburst duration [s]', 'pad':0.1})
    annotate_str = f'({string.ascii_lowercase[i]}) {percentiles[i]}th percentile'
    ax_i.text(-0.2, 1.2, annotate_str, va='top', transform=ax_i.transAxes, 
            weight='bold', fontsize=15)

d4 = dial_plot.Dial(ax[-1], MLT_bins, L_bins, num_microbursts_H)
d4.draw_dial(L_labels=L_labels,
            mesh_kwargs={'norm':matplotlib.colors.LogNorm()},
            colorbar_kwargs={'label':'Number of microbursts', 'pad':0.1})
annotate_str = f'({string.ascii_lowercase[len(ax)-1]}) Microburst occurrence'
ax[-1].text(-0.2, 1.2, annotate_str, va='top', transform=ax[-1].transAxes, 
            weight='bold', fontsize=15)

for ax_i in ax:
    ax_i.set_rlabel_position(235)
plt.suptitle(f'Distribution of SAMPEX microburst durations in L-MLT', fontsize=20)

plt.tight_layout()
plt.show()