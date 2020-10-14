import pathlib
from datetime import datetime, date

import numpy as np
import pandas as pd

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import load_hilt_data

class Append_Attitude:
    def __init__(self, catalog_path):
        self.catalog_path = catalog_path
        self.attitude_keys = ['GEO_Long', 'GEO_Lat', 'Altitude', 'L_Shell', 'MLT']

        self.load_catalog()
        return
    
    def loop(self):
        """

        """
        #self.prev_date = pd.Timestamp.min
        self.attitude_df = pd.DataFrame(index=self.catalog.index) 
        catalog_copy = self.catalog.copy() # Keep the original to apply the merge.
        self.catalog[self.attitude_keys] = np.nan

        for unique_date in self.unique_dates:
            # Load attitude data if it has not been loaded yet,
            # or if the unique_date is not in the file (then 
            # load the next attitude file)
            if ((not hasattr(self, 'attitude')) or 
                (self.attitude.attitude[self.attitude.attitude.index.date == unique_date].shape[0] == 0)):
                print(f'Loading attitude file for {unique_date}')
                try:
                    self.attitude = load_hilt_data.Load_SAMPEX_Attitude(unique_date)
                except ValueError as err:
                    # The last day doesn't have attitude data so skip it.
                    if str(err) == 'A matched file not found for year=2012, doy=311':
                        print(err)
                        continue
            self.merged = pd.merge_asof(catalog_copy, self.attitude.attitude, left_index=True, 
                                right_index=True, tolerance=pd.Timedelta(seconds=10),
                                direction='nearest')
            self.catalog.update(self.merged)
        return

    def load_catalog(self):
        """
        Load the SAMPEX catalog and parse the datetime column.
        """
        self.catalog = pd.read_csv(self.catalog_path, 
            index_col=0, parse_dates=True)
        self.catalog['date'] = self.catalog.index.date
        
        # for attitude_key in attitude_keys:
        #     self.catalog[attitude_key] = np.nan
        self.unique_dates = self.catalog.date.unique()
        return

    def save_catalog(self):
        """

        """
        self.catalog.to_csv(self.catalog_path, index_label='dateTime')


if __name__ == "__main__":
    cat_path = pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_00.csv')
    a = Append_Attitude(cat_path)
    a.loop()
    a.save_catalog()