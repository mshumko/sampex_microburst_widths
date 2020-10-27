# Figure 1: N examples of microbursts and their fits.
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker

from sampex_microburst_widths import config
from sampex_microburst_widths.microburst_id.identify_microbursts import SAMPEX_Microburst_Widths
from sampex_microburst_widths.misc import load_hilt_data

plt.rcParams.update({'font.size': 15})

count_rate_conversion=int(1/20E-3)
yaxis_scale_factor=10000

catalog_name = 'microburst_catalog_02.csv'
cat = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name), 
                index_col=0, parse_dates=True)
cat['t0'] = pd.to_datetime(cat['t0'])

random=False
times = pd.to_datetime([
    '1999-11-08 02:45:30.320000',
    '1997-11-09 19:57:09.720000',
    '2000-10-29 10:45:23.180000',
    # '2001-05-09 04:03:30.100000',
    '2003-06-28 17:25:07.320000',
    # '2005-08-31 19:23:00.020000'
    ]).sort_values()

plot_width_s = 2
plot_half_width = pd.Timedelta(seconds=plot_width_s/2)

if random:
    n = 5
    plot_df = cat.sample(n=n, replace=False).sort_index()
else:
    n = len(times)
    plot_df = cat.loc[times, :]

fig, ax = plt.subplots(1, n, figsize=(12, 4))

for label_i, ax_i, (row_time, row) in zip(string.ascii_lowercase, ax, plot_df.iterrows()):
    print(label_i, row_time)
    time_range = [row_time-plot_half_width, row_time+plot_half_width]
    start_time_sec_floor = time_range[0].replace(second=0, microsecond=0)
    
    # Load and plot the HILT data
    hilt_data = load_hilt_data.Load_SAMPEX_HILT(row_time)
    hilt_data.resolve_counts_state4()
    hilt_filtered = hilt_data.hilt_resolved.loc[time_range[0]:time_range[1], :]
    ax_i.plot(hilt_filtered.index, 
            (count_rate_conversion/yaxis_scale_factor)*hilt_filtered.counts, 
            c='k')

    # Plot the fit
    current_date = hilt_filtered.index[0].floor('d')
    time_seconds = (hilt_filtered.index-current_date).total_seconds()
    popt = row.loc[['A', 't0', 'fwhm', 'y-int', 'slope']]
    popt[1] = (popt[1] - current_date).total_seconds()
    popt[2] = popt[2]/2.355 # Convert the Gaussian FWHM to std
    y = SAMPEX_Microburst_Widths.gaus_lin_function(time_seconds, *popt)
    ax_i.plot(hilt_filtered.index, (count_rate_conversion/yaxis_scale_factor)*y, 
            c='r', ls='--')

    # Format the time axis as seconds.
    ax_i.xaxis.set_minor_locator(mdates.MicrosecondLocator(interval=100_000))
    ax_i.xaxis.set_major_locator(mdates.SecondLocator())
    # Get the axis ticks first
    prev_ticks = mdates.num2date(ax_i.get_xticks())
    prev_ticks = [tick_i.replace(tzinfo=None) for tick_i in prev_ticks]
    sec_ticks = [int((tick_i - start_time_sec_floor).total_seconds()) for tick_i in prev_ticks]
    ax_i.set_xticklabels(sec_ticks)
    # Set the yticks to be at integer values.
    ax_i.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    annotate_str = (
                f'({label_i})\n'
                f'fwhm = {round(row.fwhm, 2)} [s]\n'
                f'r2 = {round(row.r2, 2)}\n'
                #f'adj_r2 = {round(row.adj_r2, 2)}'
                )
    ax_i.text(0, 1, annotate_str, va='top', ha='left', transform=ax_i.transAxes)
    min_max = [0.9*(count_rate_conversion/yaxis_scale_factor)*hilt_filtered.counts.min(), 
                1.1*(count_rate_conversion/yaxis_scale_factor)*hilt_filtered.counts.max()]
    ax_i.set(xlabel=f'seconds after\n{start_time_sec_floor}', ylim=min_max)

plt.suptitle('SAMPEX/HILT Microburst Fits')
if yaxis_scale_factor > 1:
    ax[0].set_ylabel(f'{yaxis_scale_factor} x counts/s')    
else:
    ax[0].set_ylabel(f'counts/s')    

plt.subplots_adjust(left=0.08, right=0.99, wspace=0.2, bottom=0.20, top=0.92)
plt.show()