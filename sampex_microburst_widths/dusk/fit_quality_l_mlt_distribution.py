"""
Plot the L-MLT distribution of the number of good and bad quality fits.
"""
import string
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors

from sampex_microburst_widths import config
from sampex_microburst_widths.stats import dial_plot


catalog_name = 'microburst_catalog_04.csv'
good_r2_thresh = 0.9
bad_r2_thresh = 0.5
L_bins = np.linspace(2, 8.1, num=20)
L_labels = [2,4,6]
MLT_bins = np.linspace(0, 24, num=40)
cmap = 'viridis'

catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', catalog_name)
catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)
catalog['width_ms'] = 1000*catalog['width_s'] # Convert seconds to ms.
catalog['fwhm_ms'] = 1000*catalog['fwhm']
catalog['fwhm_ms'] = catalog['fwhm_ms'].abs()

good_fit_catalog = catalog.loc[catalog['adj_r2'] > good_r2_thresh, :]
bad_fit_catalog = catalog.loc[catalog['adj_r2'] < bad_r2_thresh, :]

fig = plt.figure(figsize=(7, 4))
ax = [plt.subplot(1, 2, i, projection='polar') for i in range(1, 3)]

good_H, _, _ = np.histogram2d(good_fit_catalog.loc[:, ['MLT', 'L_Shell']], MLT_bins, L_bins)
d = dial_plot.Dial(ax[0], MLT_bins, L_bins, good_H.T)
d.draw_dial(L_labels=L_labels,
        mesh_kwargs={'cmap':cmap, 'norm':matplotlib.colors.Normalize(60, 150)},
        colorbar_kwargs={'label':f'microburst duration [ms]', 'pad':0.1})
annotate_str = f'({string.ascii_lowercase[0]}) adJ_R^2 > {good_r2_thresh}'
ax[0].text(-0.2, 1.2, annotate_str, va='top', transform=ax[0].transAxes, 
        weight='bold', fontsize=15)

plt.show()