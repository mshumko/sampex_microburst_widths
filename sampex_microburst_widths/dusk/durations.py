"""
Plot >1 MeV microburst durations in the dusk MLT range.
"""
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config

catalog_name = 'microburst_catalog_04.csv'
mlt_range = [6, 12]
good_r2_thresh = 0.9
bad_r2_thresh = 0.5

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

fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)

ax[0].hist(good_fit_catalog['fwhm_ms'], bins=np.linspace(0, 1000))
ax[0].text(0.98, 0.98, f'$R^{{2}} > {{{good_r2_thresh}}}$', 
           ha='right', va='top', transform=ax[0].transAxes)
ax[1].hist(bad_fit_catalog['fwhm_ms'], bins=np.linspace(0, 1000))
ax[1].text(0.98, 0.98, f'$R^{{2}} < {{{bad_r2_thresh}}}$', 
           ha='right', va='top', transform=ax[1].transAxes)
ax[0].set(ylabel='number of microbursts', title=f'>1 MeV microburst durations | {mlt_range=}')
ax[1].set(ylabel='number of microbursts', xlabel='FWHM [ms]', xlim=(0,1000))
ax[0].set_yscale('log')
ax[1].set_yscale('log')
plt.tight_layout()
plt.show()