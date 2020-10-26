# Figure 1: N examples of microbursts and their fits.
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator
from sampex_microburst_widths.misc import load_hilt_data

catalog_name = 'microburst_catalog_02.csv'
cat = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name), 
                index_col=0, parse_dates=True)

random=True
n = 3
plot_width_s = 5
plot_half_width = pd.Timedelta(seconds=plot_width_s/2)

plot_df = cat.sample(n=5, random_state=124, replace=False).sort_index()

fig, ax = plt.subplots(1, n, figsize=(10, 4))

for label_i, ax_i, (row_time, row) in zip(string.ascii_lowercase, ax, plot_df.iterrows()):
    print(row_time)
    time_range = [row_time-plot_half_width, row_time+plot_half_width]
    # Load the data
    hilt_data = load_hilt_data.Load_SAMPEX_HILT(row_time)
    hilt_data.resolve_counts_state4()
    hilt_filtered = hilt_data.hilt_resolved.loc[time_range[0]:time_range[1], :]

    ax_i.plot(hilt_filtered.index, hilt_filtered.counts, c='k')

    ax_i.set_xlabel('UTC')
    ax_i.text(0, 1, f'({label_i})', va='top', ha='left', transform=ax_i.transAxes)

ax[0].set_ylabel('Counts/s')    
fig.tight_layout()
plt.show()