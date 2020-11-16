import pathlib
import re
import subprocess
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num, num2date
import progressbar
import scipy.optimize
import scipy.signal
import sklearn.metrics

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import load_hilt_data
from sampex_microburst_widths.microburst_id import signal_to_background

class Identify_SAMPEX_Microbursts:
    def __init__(self, baseline_width_s=0.500, foreground_width_s=0.1,
                threshold=10, spin_file_name='spin_times.csv', 
                prominence_rel_height=0.5):
        self.hilt_dir = pathlib.Path(config.SAMPEX_DIR, 'hilt', 'State4')
        self.spin_file_name = spin_file_name
        self.baseline_width_s = baseline_width_s
        self.foreground_width_s = foreground_width_s
        self.threshold = threshold
        self.prominence_rel_height = prominence_rel_height
        return

    def loop(self, debug=False):
        """
        Loop over the HILT files and run the following steps:
        - get the date from the file name,
        - verify that SAMPEX was not in spin mode,
        - load and resolve the 20 ms HILT file,
        - Run the microburst detection code on it, and
        - save to self.microbursts.
        """
        self.get_file_names()
        self.load_spin_times()
        self.microburst_times = pd.DataFrame(data=np.zeros((0, 2)), 
                                            columns=['dateTime', 'burst_param'])

        for hilt_file in progressbar.progressbar(self.hilt_files, redirect_stdout=True):
            date = self.get_filename_date(hilt_file)

            # Skip if the file name date was during the SAMPEX spin.
            # OR the data is from 1996
            if self.date_during_spin(date) or (date.year == 1996):
                continue

            # Load the data
            try:
                self.hilt_obj = load_hilt_data.Load_SAMPEX_HILT(date)
            except RuntimeError as err:
                if str(err) == "The SAMPEX HITL data is not in order.":
                    continue
                else:
                    raise
            # Resolve the 20 ms data
            self.hilt_obj.resolve_counts_state4()

            # Use the hilt data to id microbursts  
            try:
                self.id_microbursts(debug=debug)
            except ValueError as err:
                if str(err) == 'No detections found':
                    print(err, hilt_file.name)
                    continue
                else:
                    raise
            # self.test_detections()
        return

    def get_file_names(self):
        """
        Get a sorted list of file names in the State4 directory
        """
        hilt_files_generator = self.hilt_dir.glob('*')
        self.hilt_files = sorted(list(hilt_files_generator))
        if len(self.hilt_files) == 0:
            raise FileNotFoundError('No HILT files found. Is the data directory avaliable '
                                    'and defined in config.SAMPEX_DIR?')
        return

    def date_during_spin(self, date):
        """ 
        This method checks if the file name taken on date
        was taken when SAMPEX was in the spin mode.
        """
        # This returns True if 
        in_spin_time = ((date >= self.spin_times['start']) & 
                        (date <= self.spin_times['end']))
        return any(in_spin_time) # True if the date was during the spin time.

    def load_spin_times(self):
        """ Load and parse the spin time file """
        spin_file_path = pathlib.Path(config.PROJECT_DIR, 'data', self.spin_file_name)
        self.spin_times = pd.read_csv(spin_file_path)
        for column in self.spin_times.columns:
            # Remove the time of day that SAMPEX entered spin mode 
            # since we will exclide the entire day.
            self.spin_times[column] = self.spin_times[column].apply(
                                        lambda s: s.split('T')[0])
            self.spin_times[column] = pd.to_datetime(self.spin_times[column])
        return

    def get_filename_date(self, file_path):
        """ Given a filename find the date using regex and pd.to_datetime"""
        file_name = file_path.name
        # Pick off the numbers out of the filename.
        year_doy_str = re.findall(r'\d+', str(file_name))[0]
        # Parse the date assuming a YYYYDOY format.
        return pd.to_datetime(year_doy_str, format='%Y%j')

    def id_microbursts(self, debug=False):
        """ Use SignalToBackground class to identify microbursts """
        self.stb = signal_to_background.SignalToBackground(
                                    self.hilt_obj.counts, 20E-3, 
                                    self.baseline_width_s,
                                    foreground_width_s=self.foreground_width_s)
        self.stb.significance()
        self.stb.find_microburst_peaks(std_thresh=self.threshold)

        # Remove detections made near data gaps (where the baseline is invalid)
        self.remove_detections_near_time_gaps()

        # Calculate the microburst widths using the prominence method and
        # the Gaussian fit.
        gaus = SAMPEX_Microburst_Widths(self.hilt_obj.hilt_resolved, self.stb.peak_idt)
        gaus.calc_prominence_widths(self.prominence_rel_height)
        fit_df = gaus.calc_gaus_widths(debug=debug)

        # Save to a DataFrame
        df = pd.DataFrame(
            data={
                'dateTime':self.hilt_obj.hilt_resolved.iloc[self.stb.peak_idt, :].index,
                'width_s':gaus.prom_widths_s,
                'width_height':gaus.width_height,
                'left_peak_base':gaus.left_peak_base,
                'right_peak_base':gaus.right_peak_base,
                'burst_param':np.round(self.stb.n_std.values[self.stb.peak_idt].flatten(), 1)
                },
            index=self.stb.peak_idt
            )
        merged_df = pd.concat([df, fit_df], axis=1)
        self.microburst_times = self.microburst_times.append(merged_df)
        return

    def remove_detections_near_time_gaps(self):
        """

        """
        times = self.hilt_obj.hilt_resolved.index
        dt = (times[1:] - times[:-1]).total_seconds()
        bad_indices = np.array([])
        bad_index_range = int(5/(dt[0]*2))
        # Loop over every peak and check that the nearby data has no
        # time gaps longer than 1 second.
        for i, peak_i in enumerate(self.stb.peak_idt):
            if dt[peak_i-bad_index_range:peak_i+bad_index_range].max() > 1:
                bad_indices = np.append(bad_indices, i)
        self.stb.peak_idt = np.delete(self.stb.peak_idt, bad_indices.astype(int))
        return

    def save_catalog(self, save_name=None):
        """ 
        Saves the microburst_times DataFrame to a csv file 
        with the save_name filename. If save_name is none,
        a default catalog name will be used:
        'microburst_catalog_###.csv', where ### is a verion 
        counter, starting from 0. If a filename already exists,
        the counter increments and checks if that filename 
        already exists.

        This method also saves the creation time, catalog name,
        and git revision hash to catalog_log.csv also in the
        data subdirectory.
        """
        # Drop duplicate detections, if any
        pre_len = self.microburst_times.shape[0]
        self.microburst_times.drop_duplicates(subset='dateTime', inplace=True)
        print(f'{pre_len - self.microburst_times.shape[0]} duplicate detections dropped.')
        # If the save_name is None, save to a default filename
        # that gets incremented if it already exists.
        if save_name is None:
            counter = 0
            while True:
                save_path = pathlib.Path(config.PROJECT_DIR, 'data',
                    'microburst_catalog_{:02d}.csv'.format(counter))
                if not save_path.exists():
                    break
                counter += 1
        else:
            save_path = pathlib.Path(config.PROJECT_DIR, 'data', save_name)

        # Save the microburst catalog
        log_path = pathlib.Path(config.PROJECT_DIR, 'data', 'catalog_log.csv')
        self.microburst_times.to_csv(save_path, index=False)

        # Log the saved catalog info.
        git_revision_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD']
            ).strip().decode()
        log = pd.DataFrame(
            index=[0],
            data={ 
                'time':pd.Timestamp.today(),
                'catalog_name':save_path.name,
                'git_revision_hash':git_revision_hash
                })
        # Determine if the header needs to be written
        if log_path.exists():
            header=False
        else:
            header=True
        log.to_csv(log_path, 
                mode='a', header=header, index=False)
        return save_path

    def test_detections(self):
        """ This method plots the microburst detections """
        # plt.plot(pd.Series(self.hilt_obj.times), self.hilt_obj.counts, 'k') 
        plt.plot(pd.Series(self.hilt_obj.times), self.hilt_obj.counts, 'k') 
        plt.scatter(pd.Series(self.hilt_obj.times[self.stb.peak_idt]), 
                    self.hilt_obj.counts[self.stb.peak_idt], 
                    c='r', marker='D')
        plt.show()

class SAMPEX_Microburst_Widths:
    def __init__(self, hilt_data, peak_idt, width_multiplier=2, plot_width_s=5):
        """
        
        """
        self.hilt_data = hilt_data
        self.hilt_times = self.hilt_data.index.to_numpy()
        self.peak_idt = peak_idt
        self.width_multiplier = width_multiplier
        self.plot_width_s = plot_width_s
        return

    def calc_prominence_widths(self, prominence_rel_height=0.5):
        """
        Use scipy to find the peak width at self.prominence_rel_height prominence
        """
        # Check that self.stb.peak_idt correspond to the max values
        peak_check_thresh = 5 # Look 100 ms around the peak count to find the true peak. 
        for i, index_i in enumerate(self.peak_idt):
            self.peak_idt[i] = index_i - peak_check_thresh + \
                np.argmax(self.hilt_data['counts'][index_i-peak_check_thresh:index_i+peak_check_thresh])

        # Use scipy to find the peak width at self.prominence_rel_height prominence
        widths_tuple = scipy.signal.peak_widths(self.hilt_data['counts'], self.peak_idt, 
                                            rel_height=prominence_rel_height)
        self.prom_widths_s = 20E-3*widths_tuple[0]   
        self.width_height = widths_tuple[1]
        self.left_peak_base = self.hilt_times[np.round(widths_tuple[2]).astype(int)]
        self.right_peak_base = self.hilt_times[np.round(widths_tuple[3]).astype(int)]                                 
        return self.prom_widths_s, self.width_height, self.left_peak_base, self.right_peak_base

    def calc_gaus_widths(self, debug=False, detrend=True):
        """

        """
        if not hasattr(self, 'prom_widths_s'):
            raise AttributeError('No prior width estimate exists. Run the '
                                'calc_prominence_widths method first.')

        # Create empty pd.DataFrames for fit variables.
        fit_param_names = ['r2', 'adj_r2', 'A', 't0', 'fwhm']
        if detrend:
            fit_param_names.extend(['y-int', 'slope'])
        df = pd.DataFrame(data={key:np.nan*np.ones_like(self.peak_idt) 
                        for key in fit_param_names}, index=self.peak_idt)
        
        # Loop over every peak
        for i, (peak_i, width_i, height_i) in enumerate(zip(self.peak_idt, self.prom_widths_s, self.width_height)):
            time_range = [
                self.hilt_times[peak_i]-pd.Timedelta(seconds=width_i)*self.width_multiplier,
                self.hilt_times[peak_i]+pd.Timedelta(seconds=width_i)*self.width_multiplier
                        ]
            # If too little data points, assume a 500 ms fit width.
            if len(self.hilt_data.loc[time_range[0]:time_range[1], :].index) < 5:
                time_range = [
                            self.hilt_times[peak_i]-pd.Timedelta(seconds=0.25),
                            self.hilt_times[peak_i]+pd.Timedelta(seconds=0.25)
                        ]
            t0 = self.hilt_times[peak_i]

            if width_i < 0.1:
                # If the prominence method width is small 
                # change it to a 0.1 s width as a starting guess.
                width_i = 0.1 

            if detrend:
                p0 = [
                    height_i,   # gauss amplitude 
                    t0,         # gauss center time
                    width_i,    # 2x gaus std.
                    self.hilt_data.loc[time_range[0]:time_range[1], 'counts'].median(), # y-intercept
                    0           # Slope
                    ]
            else:
                p0 = [height_i, t0, width_i]

            # Catch warnings
            with warnings.catch_warnings(record=True) as w:
                # Catch exceptions
                try:
                    popt, pcov, r2, adj_r2 = self.fit_gaus(time_range, p0)
                except RuntimeError as err:
                    if ('Optimal parameters not found: Number of calls '
                        'to function has reached maxfev') in str(err):
                        continue
                    raise
                if len(w):
                    print(w[0].message, '\n', p0, popt)
            
            # Save to a pd.DataFrame row.
            df.iloc[i, :2] = r2, adj_r2
            df.iloc[i, 2:] = popt 
            if debug:
                self.fit_test_plot(t0, time_range, popt, r2, adj_r2)
        return df

    def fit_gaus(self, time_range, p0):
        """
        Fits a gausian shape with an optinal linear detrending term.
        """
        x_data = self.hilt_data.loc[time_range[0]:time_range[1], :].index
        current_date = x_data[0].floor('d')
        x_data_seconds = (x_data-current_date).total_seconds()
        y_data = self.hilt_data.loc[time_range[0]:time_range[1], 'counts']

        if len(x_data) < len(p0):
            raise ValueError('Not enough data points to fit. Increase the '
                            'time_range or self.width_multiplier')

        p0[0] *= 2
        p0[1] = (p0[1] - current_date).total_seconds()
        p0[2] = p0[2]/2 # Convert the microburst width guess to ~std.

        popt, pcov = scipy.optimize.curve_fit(SAMPEX_Microburst_Widths.gaus_lin_function, 
                                                x_data_seconds, y_data, p0=p0, maxfev=5000)
        popt_np = -1*np.ones(len(popt), dtype=object)
        popt_np[0] = popt[0]
        popt_np[1] = current_date + pd.Timedelta(seconds=float(popt[1]))
        popt_np[2] = (2*np.sqrt(2*np.log(2)))*popt[2]
        if len(popt) == 5:
            # If superposed a Gaussian on a linear trend...
            popt_np[3:] = popt[3:]

        y_pred = SAMPEX_Microburst_Widths.gaus_lin_function(x_data_seconds, *popt)
        try:
            r2, adj_r2 = self.goodness_of_fit(y_data, y_pred, len(popt))
        except ValueError as err:
            if 'Input contains NaN, infinity or a value too large' in str(err):
                print(f'popt={popt}')
                print(f'y-data={y_data}')
                print(f'y_pred={y_pred}')
            raise
        return popt_np, np.sqrt(np.diag(pcov)), r2, adj_r2

    @staticmethod
    def gaus_lin_function(t, *args):
        """
        Args is an array of either 3 or 5 elements. First three elements are
        the Guassian amplitude, center time, and width. The last two optional
        elements are the y-intercept and slope for a linear trend. 
        """
        exp_arg = -(t-args[1])**2/(2*args[2]**2)
        y = args[0]*np.exp(exp_arg)

        if len(args) == 5:
            y += args[3] + t*args[4]
        return y

    def fit_test_plot(self, peak_time, time_range, popt, r2, adj_r2, ax=None):
        """
        Make a test plot of the microburst fit and annotate the fit 
        parameters.
        """
        if ax is None:
            _, ax = plt.subplots()
        plot_time_range = [
            peak_time - pd.Timedelta(seconds=self.plot_width_s/2),
            peak_time + pd.Timedelta(seconds=self.plot_width_s/2)
        ]

        time_array = self.hilt_data.loc[plot_time_range[0]:plot_time_range[-1]].index
        current_date = time_array[0].floor('d')
        x_data_seconds = (time_array-current_date).total_seconds()
        y_data = self.hilt_data.loc[plot_time_range[0]:plot_time_range[1], 'counts']

        popt[1] = (popt[1] - current_date).total_seconds()
        popt[2] = popt[2]/2.355 # Convert the Gaussian FWHM to std

        gaus_y = SAMPEX_Microburst_Widths.gaus_lin_function(x_data_seconds, *popt)
        ax.plot(time_array, y_data, c='k')
        ax.plot(time_array, gaus_y, c='r')

        for t_i in time_range:
            ax.axvline(t_i, c='g')

        s = (f'R^2 = {round(r2, 2)}\n'
            f'adj R^2 = {round(adj_r2, 2)}\n'
            f'A = {round(popt[0])} [counts]\n'
            f't0 = {round(popt[1])} [sec_of_day]\n'
            f'FWHM = {round(popt[2]*2.355, 2)} [s]')
        if len(popt) == 5:
            s += (f'\ny-intercept = {round(popt[3])}\n'
                  f'slope = {round(popt[4])}')
        ax.text(0, 0.95, s, va='top', ha='left', transform=ax.transAxes, fontsize=12)

        ax.set(title=f'SAMPEX microburst fit\n{peak_time}', ylim=(y_data.min(), 1.1*y_data.max()))
        plt.show()
        return

    def goodness_of_fit(self, y_true, y_pred, n_params):
        """
        Method to calculate the R^2 coefficient of determination
        and the adjusted R^2 coefficient given the number
        of fit parameters n_params.
        """
        r2 = sklearn.metrics.r2_score(y_true, y_pred)
        n = len(y_true)
        adj_r2 = 1 - (1-r2)*(n-1)/(n-1-n_params)
        return r2, adj_r2