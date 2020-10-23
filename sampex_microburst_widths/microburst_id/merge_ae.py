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
        return

    def loop(self):
        """

        """
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
        Saves the merged catalog.
        """
        self.catalog.to_csv(self.catalog_path, index_label='dateTime')