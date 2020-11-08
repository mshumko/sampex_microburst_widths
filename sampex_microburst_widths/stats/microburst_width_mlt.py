# Histogram the microburst widths as a function of MLT
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

mlt_bins={'nightside':[21, 3], #'dawnside':[3, 9], 
          'dayside':[9, 15]}#, 'duskside':[15, 21]}
max_width=0.5
width_bins=np.arange(0, max_width, 0.02)

r2_thresh = 0.9
df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_02.csv'))

if hasattr(df, 'fwhm'):
    df['fwhm'] = df['fwhm'].abs()
    # Filter by the R^2 value with the microburst FWHM 
    df = df[(df['adj_r2'] > r2_thresh) & (df['fwhm'] < max_width)]
else:
    df = df[df['width_s'] < max_width]

mlt_df = {}
for label, mlt_range in mlt_bins.items():
    if label != 'nightside':
        mlt_df[label] = df[(df['MLT'] > mlt_range[0]) & ((df['MLT'] < mlt_range[1]))]
    else:
        mlt_df[label] = df[(df['MLT'] > mlt_range[0]) | ((df['MLT'] < mlt_range[1]))]

_, ax = plt.subplots(len(mlt_bins), 1, sharex=True, sharey=True, figsize=(4, 6))

for ax_i, (mlt_label, mlt_vals) in zip(ax, mlt_df.items()):
    ax_i.hist(mlt_vals.loc[:, 'fwhm'].to_numpy(), bins=width_bins, density=True)
    ax_i.text(1, 1, f'median={round(mlt_vals.loc[:, "fwhm"].median(), 2)} [s]', 
            va='top', ha='right', transform=ax_i.transAxes)
    ax_i.set_title(mlt_label)
    ax_i.set_ylabel('probability density')

ax[-1].set(xlim=(0, max_width), ylim=(0, None), xlabel='Microburst FWHM [s]')

# # df.plot(x='MLT', y='fwhm', kind='hexbin', ax=ax)
# H, _, _ = np.histogram2d(df['MLT'], df['fwhm'], bins=(MLT_bins, width_bins))
# p = ax.pcolormesh(MLT_bins, width_bins, H.T)
# plt.colorbar(p, ax=ax)
# ax.set_xlim(0, 24)
# ax.set_ylim(0, max_width)
# # df['fwhm'].plot.hist(ax=ax, bins=np.linspace(0, max_width))
# ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 s, a_500=0.5 s"
#                     f"\nGaussian FWHM | min R^2 = {r2_thresh}"
#                     "\nno visual inspection", 
#         ha='right', va='top', transform=ax.transAxes
#         )

# MLT_bins = np.arange(0, 25, 3)
# width_bins = np.linspace(0, 0.25)
# MLT_binned = [ 
#             df.loc[(df['MLT'] > l_MLT) & (df['MLT'] < u_MLT), 'width_s'].to_numpy()
#             for l_MLT, u_MLT in zip(MLT_bins[:-1], MLT_bins[1:])
#             ]

# _, bx = plt.subplots(len(MLT_bins)-1, 1, figsize=(5,8), sharex=True, sharey=True)
# for bx_i, MLT_array, MLT_i, MLT_f in zip(bx, MLT_binned, MLT_bins, MLT_bins[1:]):
#     # Calc percentiles
#     q = np.percentile(MLT_array, [25, 50, 75])
#     bx_i.hist(MLT_array, bins=width_bins)
#     for q_i in q:
#         bx_i.axvline(q_i, c='k')
#     bx_i.text(0, 1, f"{MLT_i} < MLT < {MLT_f}", ha='left', va='top', transform=bx_i.transAxes)
# bx[-1].set_xlabel('Width [s]')
# obx[0].set_title('SAMPEX microburst widths')

plt.tight_layout()
plt.show()