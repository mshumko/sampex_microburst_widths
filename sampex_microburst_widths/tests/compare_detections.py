import pathlib
import re

import pandas as pd
import scipy.io
import matplotlib.pyplot as plt
import numpy as np

from sampex_microburst_widths.misc import load_hilt_data
try:
    from sampex_microburst_widths import config
except ModuleNotFoundError:
    print('The config.py file is not found. Run "python3 -m sampex_microburst_widths init"')
    raise

class CompareDetections:
    def __init__(self, thresh_s=0.1):
        """
        This class handles the loading and comparing of 
        Lauren's microburst database with mine.
        """
        self.thresh_s = thresh_s
        return

    def load_test_data(self, test_dir='tests'):
        """
        Load Lauren's test data and put it into one DataFrame.
        """
        files_path_gen = pathlib.Path(config.PROJECT_DIR, test_dir).glob('*.mat')
        files_path_list = sorted(list(files_path_gen))
        array_list = len(files_path_list)*[None]

        for i, file in enumerate(files_path_list):
            array_list[i] = self.load_mat_file(file)
        self.mat_times = pd.concat(array_list)
        return

    def load_microburst_catalog(self, catalog_name):
        """

        """
        cat_path = pathlib.Path(config.PROJECT_DIR, 'data', catalog_name)
        self.cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
        return

    def merge_catalogs(self):
        self.merged_df = pd.merge_asof(self.mat_times, self.cat, 
        left_on='dateTime', right_index=True, direction='nearest', 
        tolerance=pd.Timedelta(self.thresh_s, unit='s'))
        return

    def plot_detections(self, date='1999-05-02'):
        """ Plots Lauren's and my detections on top of the HILT data """
        l = load_hilt_data.Load_SAMPEX_HILT(date)
        l.resolve_counts_state4()
        # a = load_hilt_data.Load_SAMPEX_Attitude(date)

        filtered_mat_times = self.mat_times.loc[date].index
        filtered_cat_times = self.cat.loc[date].index

        plt.plot(l.hilt_resolved.index[::10], l.hilt_resolved.counts[::10])
        plt.scatter(filtered_mat_times, 10*np.ones(filtered_mat_times.shape[0]), c='r')
        plt.scatter(filtered_cat_times, 15*np.ones(filtered_cat_times.shape[0]), c='b')
        plt.yscale('log')
        plt.show()
        return

    def load_mat_file(self, file_path):
        """
        Loads the matlab data times and returns a pandas Timestamp array.
        """
        mat_data = scipy.io.loadmat(file_path)
        date = self.get_year_from_filename(file_path)
        burst_index = mat_data['burstindex'][0, :]
        burst_hr = mat_data['time_hr'][burst_index, 0]
        df = pd.DataFrame(date + pd.to_timedelta(burst_hr, unit='hr'), 
                        columns=['dateTime'])
        df.index = df.dateTime
        return df

    def get_year_from_filename(self, file_name):
        """
        Given the filename with the date in the format, for example,
        1998DOY292, this function finds this section of the filename
        and parses out the year and day of year and returns
        a pandas Timestamp object.
        """
        match = re.search(r'\d{4}DOY\d{3}', str(file_name)).group(0)
        return pd.to_datetime(match, format="%YDOY%j")


if __name__ == '__main__':
    c = CompareDetections()
    c.load_test_data()
    c.load_microburst_catalog('microburst_test_catalog.csv')
    # c.merge_catalogs()
    c.plot_detections()