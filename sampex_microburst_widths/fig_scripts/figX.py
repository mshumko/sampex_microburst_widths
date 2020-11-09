"""
Histogram of all microburst widths.

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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

plt.rcParams.update({'font.size': 13})

### Script parameters ###
catalog_name = 'microburst_catalog_02.csv'
r2_thresh = 0.9
max_width = 0.5
width_bins = np.linspace(0, max_width+0.001, num=50)

# Load the catalog, drop the NaN values, and filter by the max_width and
# R^2 values.
df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
df.dropna(inplace=True)
df = df[df['width_s'] < max_width]
df['fwhm'] = df['fwhm'].abs()
df = df[df.adj_r2 > r2_thresh]

fig, ax = plt.subplots()
ax.hist(df['width_s'], bins=width_bins, color='k', histtype='step', density=True)
ax.set_title('Distribution of SAMPEX > 1 MeV Microburst Durations')
# ax.set_yscale('log')
ax.set_ylabel('Probability Density')
ax.set_xlabel('FWHM [s]')
# ax.text(0.95, 0.95, "O'Brien burst parameter\nWidth at half prominence\nno visual inspection", 
#         ha='right', va='top', transform=ax.transAxes
#         )
ax.set_xlim(0, max_width)

plt.show()