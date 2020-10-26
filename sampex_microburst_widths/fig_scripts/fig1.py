# Figure 1: N examples of microbursts and their fits.
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.microburst_id.identify_microbursts import SAMPEX_Microburst_Widths
# from sampex_microburst_widths.misc import plot_annotator_decorator
from sampex_microburst_widths.misc import load_hilt_data

catalog_name = 'microburst_catalog_02.csv'
cat = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name), 
                index_col=0, parse_dates=True)
cat['t0'] = pd.to_datetime(cat['t0'])

random=True
times = [
    '1999-11-08 02:45:30.320000',
    '1997-11-08 21:26:15.980000',
    '1997-11-08 23:03:08.740000',
    '1997-11-09 19:57:09.720000'
    ]
n = 5
plot_width_s = 5
plot_half_width = pd.Timedelta(seconds=plot_width_s/2)

if random:
    plot_df = cat.sample(n=5, random_state=0, replace=False).sort_index()
else:
    plot_df = cat.loc[times, :]

fig, ax = plt.subplots(1, n, figsize=(10, 4))

for label_i, ax_i, (row_time, row) in zip(string.ascii_lowercase, ax, plot_df.iterrows()):
    print(row_time)
    time_range = [row_time-plot_half_width, row_time+plot_half_width]
    # Load the data
    hilt_data = load_hilt_data.Load_SAMPEX_HILT(row_time)
    hilt_data.resolve_counts_state4()
    hilt_filtered = hilt_data.hilt_resolved.loc[time_range[0]:time_range[1], :]

    ax_i.plot(hilt_filtered.index, hilt_filtered.counts, c='k')

    # Plot the fit
    current_date = hilt_filtered.index[0].floor('d')
    time_seconds = (hilt_filtered.index-current_date).total_seconds()
    popt = row.loc[['A', 't0', 'fwhm', 'y-int', 'slope']]
    popt[0]
    popt[1] = (popt[1] - current_date).total_seconds()
    popt[2] = popt[2]/2.355 # Convert the Gaussian FWHM to std
    y = SAMPEX_Microburst_Widths.gaus_lin_function(time_seconds, *popt)
    ax_i.plot(hilt_filtered.index, y, c='r', ls='--')

    ax_i.text(0, 1, f'({label_i})', va='top', ha='left', transform=ax_i.transAxes)

    ax_i.set(xlabel='UTC', ylim=1.1*hilt_filtered.counts.agg(['min', 'max']))

ax[0].set_ylabel('Counts/s')    
fig.tight_layout()
plt.show()