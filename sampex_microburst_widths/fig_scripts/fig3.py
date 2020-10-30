# Make a plot of the microburst width as a function of L and MLT
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from sampex_microburst_widths import config

plt.rcParams.update({'font.size': 13})

catalog_name = 'microburst_catalog_02.csv'

def main():
    r2_thresh = 0.9
    max_width = 0.25
    width_bins = np.linspace(0, max_width+0.001, num=50)
    L_bins = np.linspace(2, 8.1, num=50)
    MLT_bins = np.linspace(0, 24, num=50)

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
    df.dropna(inplace=True)
    df = df[df['width_s'] < max_width]
    df['fwhm'] = df['fwhm'].abs()
    df = df[df.adj_r2 > r2_thresh]

    H_L, _, _ = np.histogram2d(df['L_Shell'], df['fwhm'],
                            bins=[L_bins, width_bins])
    H_MLT, _, _ = np.histogram2d(df['MLT'], df['fwhm'],
                            bins=[MLT_bins, width_bins])

    # fig = plt.figure(figsize=(10, 8))
    _, ax = plt.subplots(1, 2, figsize=(12, 6))

    p_L = ax[0].pcolormesh(L_bins, width_bins, H_L.T, vmin=0)
    plt.colorbar(p_L, ax=ax[0], orientation='horizontal', label='Number of microbursts')
    ax[0].set_xlabel('L shell')
    ax[0].set_ylabel('FWHM [s]')

    p_MLT = ax[1].pcolormesh(MLT_bins, width_bins, H_MLT.T, vmin=0)
    plt.colorbar(p_MLT, ax=ax[1], orientation='horizontal', label='Number of microbursts')
    ax[1].set_xlabel('MLT')
    ax[1].set_ylabel('FWHM [s]')

    for ax_i, label_i in zip(ax, string.ascii_lowercase):
        annotate_str = f'({label_i})'
        ax_i.text(0, 1, annotate_str, va='top', color='white', weight='bold', 
                    transform=ax_i.transAxes, fontsize=18)

    plt.suptitle(f'Distribution of SAMPEX microburst durations in L and MLT', fontsize=20)
    return

main()
plt.tight_layout()
plt.show()