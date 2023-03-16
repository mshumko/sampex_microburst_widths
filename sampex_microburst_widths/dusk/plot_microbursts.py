"""
Plot >1 MeV microburst durations in the dusk MLT range.
"""
import pathlib
from datetime import date, datetime

import pandas as pd
import numpy as np
import sampex
import matplotlib.pyplot as plt
import matplotlib.dates
from matplotlib.ticker import FuncFormatter

from sampex_microburst_widths import config

class Plot_Microbursts:
    def __init__(self, catalog_name:str, mlt_range:tuple=(0,24), plot_width_s:float=5, 
                 x_labels:dict=None, r2_bounds:tuple=(0.9, 1)) -> None:
        """
        Plot >1 MeV microbursts observed by SAMPEX HILT.

        Parameters
        ----------
        catalog_name: str
            The name of the catalog to load from the config.PROJECT_DIR/data directory.
        mlt_range:tuple
            A tuple of len 2, specifying the min and max MLT values to plot.
        plot_width_s:float
            The plot width in seconds.
        x_labels:dict
            The x-axis labels to include from the SAMPEX attitude files. The keys & values must
            be the strings. The key is the string that is plotted, while the corresponding value 
            must correspond to columns in the SAMPEX attitude data.  
        r2_bounds:tuple
            A tuple of len 2 to specifying the R^2 goodness of fit microbursts to plot.

        Methods
        -------
        loop()
            The main method to loop over the microburst catalog and plot all microbursts that meet 
            the r2_bounds and mlt_range.
        plot()
            Plot one event
        """
        self.catalog_name = catalog_name
        self.mlt_range = mlt_range
        self.plot_width_s = plot_width_s
        if x_labels is None:
            self.x_labels = {"L": "L_Shell", "MLT": "MLT", "Geo Lat": "GEO_Lat", "Geo Lon": "GEO_Long"}
        else:
            self.x_labels = x_labels
        self.r2_bounds = r2_bounds
        self.plot_save_dir = pathlib.Path(config.PROJECT_DIR.parent, 'plots', 'validation')
        if not self.plot_save_dir.exists():
            self.plot_save_dir.mkdir(parents=True)
            print(f'Made {self.plot_save_dir} directory')
        self._load_catalog()
        return
    
    def loop(self):
        """
        The main method to loop over the microburst catalog and plot all microbursts that meet 
        the r2_bounds and mlt_range.
        """
        self.current_date = pd.Timestamp.min
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.15, left=0.13, right=0.95)

        for time, row in self.catalog.iterrows():
            print(f'Processing SAMPEX microburst at {time=}')
            if time.date() != self.current_date:
                print(f'Loading {time.date()}')
                _hilt = sampex.HILT(time).load()
                _att = sampex.Attitude(time).load()
                self.hilt = pd.merge_asof(_hilt, _att, left_index=True, right_index=True,
                                    tolerance=pd.Timedelta(seconds=3), direction='nearest')
                self.current_date = time.date()

            self._plot_interval(time, ax, fit_info_dict=row)

            save_name =(
                f'{time:%Y%m%d_%H%M%S}_sampex_microburst_'
                f'r2_{int(10*row["adj_r2"])}_fwhm_{int(row["fwhm_ms"])}_'
                f'mlt_{int(row["MLT"])}_l_{int(row["L_Shell"])}.png'
                )
            plt.savefig(self.plot_save_dir/save_name)
            ax.clear()
        return
    
    def plot(self, time:datetime, ax:plt.Axes=None) -> plt.Axes:
        """
        Plot an interval of SAMPEX-HILT data.

        Parameters
        ----------
        time: datetime
            The center time to plot. The plot range is time +/- plot_width_s/2.
        ax: plt.Axes
            The optional subplot object to plot on.

        Returns
        -------
        plt.Axes
            The modified subplot.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(7, 6))
            fig.subplots_adjust(hspace=0.1, wspace=0.01, top=0.93, bottom=0.15, left=0.13, right=0.95)

        # Only for the interactive mode.
        ax.format_coord = lambda x, y: "{}, {}".format(
            matplotlib.dates.num2date(x).replace(tzinfo=None).isoformat(), round(y)
        )
        
        if (not hasattr(self, 'current_date')) or (time.date() != self.current_date):
            print(f'Loading {time.date()}')
            _hilt = sampex.HILT(time).load()
            _att = sampex.Attitude(time).load()
            self.hilt = pd.merge_asof(_hilt, _att, left_index=True, right_index=True,
                                tolerance=pd.Timedelta(seconds=3), direction='nearest')
            self.current_date = time.date()

        ax = self._plot_interval(time, ax)
        return ax
    
    def _plot_interval(self, time:datetime, ax:plt.Axes, fit_info:dict=None):
        """
        Plots an interval of HILT data.

        Parameters
        ----------
        time: datetime
            The center time to plot. The plot range is time +/- plot_width_s/2.
        ax: plt.Axes
            The optional subplot object to plot on.
        fit_info: dict
            Specifies the FWHM and adj_R^2 fit parameters to annotate in the upper-right
            corner.

        Returns
        -------
        plt.Axes
            The modified subplot.
        """
        plot_time_range = (time-pd.Timedelta(seconds=self.plot_width_s/2), 
                        time+pd.Timedelta(seconds=self.plot_width_s/2))
        hilt_flt = self.hilt.loc[plot_time_range[0]:plot_time_range[1], :]

        ax.step(hilt_flt.index, hilt_flt['counts'], c='k', where="post")
        if fit_info is not None:
            ax.axvline(time, c='k', ls='--')
            annotate_str = (f'FWHM = {round(fit_info["fwhm_ms"])} [ms]\n'
                            f'$R^{{2}} = {{{round(fit_info["adj_r2"], 2)}}}$')
            ax.text(0.70, 0.98, annotate_str, 
                    ha='left', va='top', transform=ax.transAxes)
        ax.set(title=f'SAMPEX-HILT | >1 MeV Microburst Validation\n{time:%F %T}', ylabel='Counts/20 ms')
        ax.xaxis.set_major_formatter(FuncFormatter(self._format_fn))
        ax.xaxis.set_minor_locator(matplotlib.dates.SecondLocator())
        ax.set_xlabel("\n".join(["Time"] + list(self.x_labels.keys())))
        ax.xaxis.set_label_coords(-0.1, -0.02)
        return ax
    
    def _load_catalog(self):
        """
        Load a microburst catalog, convert the FWHM to milliseconds, filter by self.mlt_range,
        and filter by self.r2_bounds.
        """
        catalog_path = pathlib.Path(config.PROJECT_DIR, 'data', self.catalog_name)
        self.catalog = pd.read_csv(catalog_path, index_col=0, parse_dates=True)
        self.catalog['width_ms'] = 1000*self.catalog['width_s'] # Convert seconds to ms.
        self.catalog['fwhm_ms'] = 1000*self.catalog['fwhm']
        self.catalog['fwhm_ms'] = self.catalog['fwhm_ms'].abs()

        self.catalog = self.catalog.loc[
            ((self.catalog['MLT'] > self.mlt_range[0]) & 
             (self.catalog['MLT'] <= self.mlt_range[1])), :
             ]
        print(f'{self.catalog.shape[0]} microbursts in {self.mlt_range=}')

        self.catalog = self.catalog.loc[
            ((self.catalog['adj_r2'] > self.r2_bounds[0]) &
             (self.catalog['adj_r2'] < self.r2_bounds[1])), :
            ]
        print(f'{self.catalog.shape[0]} microbursts in {self.mlt_range=} and with {self.r2_bounds=}')
        return
    
    def _format_fn(self, tick_val, tick_pos):
        """
        The tick magic happens here. pyplot gives it a tick time, and this function 
        returns the closest label to that time. Read docs for FuncFormatter().
        """
        # Find the nearest time within 6 seconds (the cadence of the SAMPEX attitude files)
        tick_time = matplotlib.dates.num2date(tick_val).replace(tzinfo=None)
        i_min_time = np.argmin(np.abs(self.hilt.index - tick_time))
        if np.abs(self.hilt.index[i_min_time] - tick_time).total_seconds() > 6:
            raise ValueError(f"Nearest timestamp to tick_time is more than 6 seconds away")
        pd_index = self.hilt.index[i_min_time]
        # Cast np.array as strings so that it can insert the time string.
        values = self.hilt.loc[pd_index, self.x_labels.values()].to_numpy().round(2).astype(str)
        values = np.insert(values, 0, pd_index.strftime('%T.%f')[:-5])
        label = "\n".join(values)
        return label

if __name__ == '__main__':
    catalog_name = 'microburst_catalog_04.csv'
    mlt_range = [16, 22]
    plot_width_s = 120

    plotter = Plot_Microbursts(catalog_name, mlt_range, plot_width_s, r2_bounds=(-1000, .5))
    # plotter.loop()
    plotter.plot(datetime(1999, 9, 16, 11, 34, 7))
    plt.show()