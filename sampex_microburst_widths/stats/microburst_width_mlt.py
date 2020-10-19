# Histogram the microburst widths as a function of MLT
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

# @plot_annotator_decorator.annotate_plot
def main():
    max_width=0.25
    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_00.csv'))
    df = df[df['width_s'] < max_width]

    # _, ax = plt.subplots()
    # df.plot(x='MLT', y='width_s', kind='hexbin', ax=ax)
    # ax.set_xlim(0, 24)
    # ax.set_ylim(0, max_width)
    # ax.text(0.95, 0.95, "O'Brien burst parameter n_100=0.1 ms, a_500=1 s"
    #                     "\nWidth at half prominence\nno visual inspection", 
    #         ha='right', va='top', transform=ax.transAxes
    #         )

    MLT_bins = np.arange(0, 25, 3)
    width_bins = np.linspace(0, 0.25)
    MLT_binned = [ 
                df.loc[(df['MLT'] > l_MLT) & (df['MLT'] < u_MLT), 'width_s'].to_numpy()
                for l_MLT, u_MLT in zip(MLT_bins[:-1], MLT_bins[1:])
                ]

    _, bx = plt.subplots(len(MLT_bins)-1, 1, figsize=(5,8), sharex=True, sharey=True)
    for bx_i, MLT_array, MLT_i, MLT_f in zip(bx, MLT_binned, MLT_bins, MLT_bins[1:]):
        # Calc percentiles
        q = np.percentile(MLT_array, [25, 50, 75])
        bx_i.hist(MLT_array, bins=width_bins)
        for q_i in q:
            bx_i.axvline(q_i, c='k')
        bx_i.text(0, 1, f"{MLT_i} < MLT < {MLT_f}", ha='left', va='top', transform=bx_i.transAxes)
    bx[-1].set_xlabel('Width [s]')
    bx[0].set_title('SAMPEX microburst widths')
    return df

df = main()
# plt.tight_layout()
plt.show()