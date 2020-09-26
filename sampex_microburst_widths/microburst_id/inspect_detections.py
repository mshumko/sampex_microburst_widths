# This script looks at detections made on 1999-8-17, the same day 
# as the microburst plot from Douma et al., 2017
import pathlib

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import load_hilt_data

matplotlib.rcParams['agg.path.chunksize'] = 100000
date = pd.to_datetime('1999-8-17')
catalog_name = 'microburst_test_catalog.csv'
catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', catalog_name)

catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)
catalog['date'] = catalog.index.date

filtered_catalog = catalog[catalog.loc[:, 'date'] == date]

l = load_hilt_data.Load_SAMPEX_HILT(date)
l.resolve_counts_state4()
filterered_hilt = l.hilt_resolved.loc[filtered_catalog.index, :]

plt.plot(l.hilt_resolved.index, l.hilt_resolved.counts)
plt.scatter(filterered_hilt.index, filterered_hilt.counts, c='r')
plt.yscale('log')
plt.show()