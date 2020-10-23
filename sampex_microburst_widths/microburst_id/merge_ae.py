import pathlib
from datetime import datetime, date

import numpy as np
import pandas as pd

from sampex_microburst_widths import config

class Merge_AE:
    def __init__(self, catalog_path):
        """
        Append the AE index to the microburst catalog.
        """
        self.catalog_path = catalog_path
        self.load_catalog()
        return

    def loop(self):
        """

        """

        catalog_copy = self.catalog.copy() # Keep the original to apply the merge.
        self.catalog['AE'] = np.nan

        for year in self.unique_years:
            print(f'Merging AE data for year={year}')
            # Try to load the AE file.
            try:
                self.ae = self.load_ae(year)
            except AssertionError as err:
                if 'No AE files found.' in str(err):
                    print(err)
                    continue
                else:
                    raise

            merged = pd.merge_asof(catalog_copy, self.ae, left_index=True, 
                                right_index=True, tolerance=pd.Timedelta(minutes=1),
                                direction='nearest')
            self.catalog.update(merged)
        return

    def load_catalog(self):
        """
        Load the SAMPEX catalog and parse the datetime column.
        """
        self.catalog = pd.read_csv(self.catalog_path, 
            index_col=0, parse_dates=True)
        self.unique_years = self.catalog.index.year.unique()
        return

    def load_ae(self, year):
        """
        Load the AE index from the year, from the config.AE_DIR directory.
        """
        ae_paths = list(pathlib.Path(config.AE_DIR).glob(f'{year}*ae.txt'))
        assert len(ae_paths) == 1, (f'No AE files found.\nae_dir={config.AE_DIR}, '
                                    f'year={year}, ae_paths={ae_paths}')
        ae_data = pd.read_csv(ae_paths[0], sep=' ', index_col=0, 
                            parse_dates=True, comment='#', 
                            names=['dateTime', 'AE'])
        return ae_data

    def save_catalog(self):
        """
        Saves the merged catalog.
        """
        self.catalog.to_csv(self.catalog_path, index_label='dateTime')

if __name__ == "__main__":
    m = Merge_AE(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_02.csv' ))
    m.loop()
    m.save_catalog()