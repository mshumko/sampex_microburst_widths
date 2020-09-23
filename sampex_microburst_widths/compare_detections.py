import pathlib
import re

import pandas as pd
import scipy.io

import dirs

class CompareDetections:
    def __init__(self, thresh_s=0.1):
        """
        This class handles the loading and comparing of 
        Lauren's microburst database with mine.
        """
        self.thresh_s = thresh_s
        return

    def load_test_data(self, test_dir='test_data/'):
        """
        Load Lauren's test data and put it into one DataFrame.
        """
        files_path_gen = pathlib.Path(dirs.BASE_DIR, test_dir).glob('*.mat')
        files_path_list = sorted(list(files_path_gen))
        array_list = len(files_path_list)*[None]

        for i, file in enumerate(files_path_list):
            array_list[i] = self.load_mat_file(file)
        self.mat_times = pd.concat(array_list)
        return

    def load_microburst_catalog(self, catalog_name):
        """

        """
        cat_path = pathlib.Path(dirs.BASE_DIR, 'data', catalog_name)
        self.cat = pd.read_csv(cat_path, index_col=0, parse_dates=True)
        return

    def merge_catalogs(self):
        self.merged_df = pd.merge_asof(self.mat_times, self.cat, 
        left_on='dateTime', right_index=True, direction='nearest', 
        tolerance=pd.Timedelta(self.thresh_s, unit='s'))
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
    c.merge_catalogs()