"""
Histogram the microburst widths as a function of L shell
for two MLT ranges: 21-3 and 9-15 MLT.
"""
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config


catalog_name = 'microburst_catalog_02.csv'
MLT_bins = {'nightside':[21, 3], 'dayside':[9, 15]}
r2_thresh = 0.9
max_width = 0.25
width_bins=np.linspace(0, max_width, num=30)
L_bins = np.linspace(3, 8, num=30)
LL, WW = np.meshgrid(L_bins, width_bins)

df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df['fwhm'] = df['fwhm'].abs()
df = df[df.adj_r2 > r2_thresh]

mlt_H = {}
for label, MLT_bin in MLT_bins.items():
    if label == 'nightside':
        df_flt = df[(df['MLT'] > MLT_bin[0]) | (df['MLT'] < MLT_bin[1])]
    else:
        df_flt = df[(df['MLT'] > MLT_bin[0]) & (df['MLT'] < MLT_bin[1])]

    mlt_H[label], _, _ = np.histogram2d(df_flt['L_Shell'], df_flt['fwhm'], 
                                        bins=[L_bins, width_bins])

fig, ax = plt.subplots(1, 2, sharex=True, sharey=False, figsize=(10,5))

for ax_i, (label, H) in zip(ax, mlt_H.items()):
    im = ax_i.pcolormesh(LL, WW, H.T)
    fig.colorbar(im, ax=ax_i)
    # ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 s, a_500=1 s"
    #                     f"\nGaussian FWHM | min R^2 = {r2_thresh}"
    #                     "\nno visual inspection", 
    #         ha='right', va='top', transform=ax.transAxes, color='white'
    #         )
    ax_i.text(1, 0.99, f'{label} {MLT_bins[label]}', va='top', ha='right', transform=ax_i.transAxes, 
              color='white', fontsize=15)
    ax_i.set(xlabel='L_Shell', ylabel='Width [s]')

plt.suptitle(f'SAMPEX/HILT Microburst Duration-L Shell distribution\n{catalog_name}')
plt.tight_layout()
plt.show()