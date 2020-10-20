# Make a plot of the microburst width as a function of L and MLT
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator
from sampex_microburst_widths.stats import dial_plot

catalog_name = 'microburst_catalog_01.csv'

@plot_annotator_decorator.annotate_plot
def main():
    r2_thresh = 0.9
    max_width = 0.25
    # width_bins=np.linspace(0, max_width, num=50)
    L_bins = np.linspace(2, 8.1, num=20)
    L_labels = [2,4,6]
    MLT_bins = np.linspace(0, 24, num=40)

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
    df.dropna(inplace=True)
    df = df[df['width_s'] < max_width]
    df['fwhm'] = df['fwhm'].abs()
    df = df[df.r2 > r2_thresh]

    num_microbursts_H, _, _ = np.histogram2d(df['MLT'], df['L_Shell'],
                                            bins=[MLT_bins, L_bins])
    H_q25 = np.nan*np.zeros((len(MLT_bins), len(L_bins)))
    H_q50 = np.nan*np.zeros((len(MLT_bins), len(L_bins)))
    H_q75 = np.nan*np.zeros((len(MLT_bins), len(L_bins)))

    for i, (start_MLT, end_MLT) in enumerate(zip(MLT_bins[:-1], MLT_bins[1:])):
        for j, (start_L, end_L) in enumerate(zip(L_bins[:-1], L_bins[1:])):
            df_flt = df.loc[(
                (df['MLT'] > start_MLT) &  (df['MLT'] < end_MLT) &
                (df['L_Shell'] > start_L) &  (df['L_Shell'] < end_L)
                ), 'fwhm']
            if df_flt.shape[0] > 50:
                H_q25[i, j] = df_flt.quantile(0.25)
                H_q50[i, j] = df_flt.quantile(0.5)
                H_q75[i, j] = df_flt.quantile(0.75)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(221, projection='polar')
    bx = plt.subplot(222, projection='polar')
    cx = plt.subplot(223, projection='polar')
    dx = plt.subplot(224, projection='polar')


    d = dial_plot.Dial(ax, MLT_bins, L_bins, H_q25)
    d.draw_dial(L_labels=L_labels,
                colorbar_kwargs={'label':'25% microburst width [s]'})

    d2 = dial_plot.Dial(bx, MLT_bins, L_bins, H_q50)
    d2.draw_dial(L_labels=L_labels,
                colorbar_kwargs={'label':'50% microburst width [s]'})
    
    d3 = dial_plot.Dial(cx, MLT_bins, L_bins, H_q75)
    d3.draw_dial(L_labels=L_labels,
                colorbar_kwargs={'label':'75% microburst width [s]'})

    d4 = dial_plot.Dial(dx, MLT_bins, L_bins, num_microbursts_H)
    d4.draw_dial(L_labels=L_labels,
                colorbar_kwargs={'label':'Number of microbursts'})

    plt.suptitle(f'SAMPEX microburst width statistics\ncatalog_name={catalog_name}')
    return

main()
plt.show()