# This script looks at detections made on 1999-8-17, the same day 
# as the microburst plot from Douma et al., 2017
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import load_hilt_data
from sampex_microburst_widths.microburst_id import signal_to_background

matplotlib.rcParams['agg.path.chunksize'] = 100000
date = pd.to_datetime('1999-8-17')
# date = pd.to_datetime('1997-11-9')
catalog_name = 'microburst_test_catalog.csv'
catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', catalog_name)

catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)
catalog['date'] = catalog.index.date

filtered_catalog = catalog[catalog.loc[:, 'date'] == date.date()]

print(f'Plotting {filtered_catalog.shape[0]} microbursts.')

l = load_hilt_data.Load_SAMPEX_HILT(date)
l.resolve_counts_state4()

# Calculate the A500 and N100 parameters
stb = signal_to_background.SignalToBackground(
                                    l.hilt_resolved.counts, 20E-3, 
                                    0.5)
stb.significance()

filtered_hilt = l.hilt_resolved.loc[filtered_catalog.index, :]

plt.plot(l.hilt_resolved.index, l.hilt_resolved.counts)
plt.scatter(filtered_hilt.index, filtered_hilt.counts, c='r')
plt.plot(l.hilt_resolved.index, stb.rolling_background_counts)
plt.yscale('log')
plt.show()