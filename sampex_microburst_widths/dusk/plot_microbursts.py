"""
Plot >1 MeV microburst durations in the dusk MLT range.
"""
import pathlib
# from datetime import date, datetime

import pandas as pd
import numpy as np
import sampex
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

catalog_name = 'microburst_catalog_04.csv'
mlt_range = [6, 12]
good_r2_thresh = 0.9
bad_r2_thresh = 0.5
plot_width_s = 5
save_path = pathlib.Path(config.PROJECT_DIR.parent, 'plots', 'validation')

catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', catalog_name)
catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)
catalog['width_ms'] = 1000*catalog['width_s'] # Convert seconds to ms.
catalog['fwhm_ms'] = 1000*catalog['fwhm']
catalog['fwhm_ms'] = catalog['fwhm_ms'].abs()

catalog = catalog.loc[((catalog['MLT'] > mlt_range[0]) & (catalog['MLT'] <= mlt_range[1])), :]
print(f'{catalog.shape[0]} microbursts in {mlt_range=}')

good_fit_catalog = catalog.loc[catalog['adj_r2'] > good_r2_thresh, :]
bad_fit_catalog = catalog.loc[catalog['adj_r2'] < bad_r2_thresh, :]
print(f'{good_fit_catalog.shape[0]} microbursts in {mlt_range=} and with adj_R^2>{good_r2_thresh}')
print(f'{bad_fit_catalog.shape[0]} microbursts in {mlt_range=} and with adj_R^2<{bad_r2_thresh}')

current_date = pd.Timestamp.min
for time, row in good_fit_catalog.iterrows():
    print(f'Processing SAMPEX microburst at {time=}')
    if time.date() != current_date:
        print(f'Loading {time.date()}')
        _hilt = sampex.HILT(time).load()
        _att = sampex.Attitude(time).load()
        hilt = pd.merge_asof(_hilt, _att, left_index=True, right_index=True,
                               tolerance=pd.Timedelta(seconds=3), direction='nearest')
        current_date = time.date()

    plot_time_range = (time-pd.Timedelta(seconds=plot_width_s/2), 
                  time+pd.Timedelta(seconds=plot_width_s/2))
    hilt_flt = hilt.loc[plot_time_range[0]:plot_time_range[1], :]

    fig, ax = plt.subplots()
    ax[0].plot(hilt_flt.index, hilt_flt['counts'], 'k')
    annotate_str = f'FWHM = {row["fwhm_ms"]} [ms]\n$R^{{2}} = {{{row["adj_r2"]}}}$'
    ax[0].text(0.70, 0.98, annotate_str, 
           ha='left', va='top', transform=ax[0].transAxes)

    save_name = f'{time: %F %T}_sampex_microburst_good_fit.png'
    plt.savefig(save_path/'good'/save_name)