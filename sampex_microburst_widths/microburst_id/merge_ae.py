"""
This module contains the Merge_AE class that uses 
pandas.merge_asof to merge the 1 minute AE cadence
values to the data. Since this module heavily relies 
on pandas, it is very efficient and the merge takes
about 10 seconds with the AE data on an HDD.
"""

import pathlib
from datetime import datetime, date

import numpy as np
import pandas as pd

from sampex_microburst_widths import config

class Merge_AE:
    def __init__(self, catalog_path):
        """
        Append the AE index to the microburst catalog using 
        pandas.merge_asof. When this class is called the
        catalog is automatically loaded.

        Parameters
        ----------
        catalog_path: str or pathlib.Path
            The path to the catalog file.

        Returns
        -------
        None

        Example
        -------
        m = Merge_AE(pathlib.Path(
            config.PROJECT_DIR, 'data', 'microburst_catalog_02.csv'
                    ))
        m.loop()
        m.save_catalog()    
        """
        self.catalog_path = catalog_path
        self.load_catalog()
        return

    def loop(self):
        """
        Loop over the years in the catalog and merge the AE indices 
        that are within 1 minute. 

        This method creates a copy of the catalog as a reference that
        has no AE column and uses it to merge_asof with the AE index df.
        The updated catalog_copy with the AE index is then used to update
        the values in self.catalog.

        Parameters
        ----------
        None

        Returns
        -------
        None
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
        Load the SAMPEX catalog, parse the datetime column into datetime 
        objects, and creates a list of unique years to determine what 
        AE years to load.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.catalog = pd.read_csv(self.catalog_path, 
            index_col=0, parse_dates=True)
        self.unique_years = self.catalog.index.year.unique()
        return

    def load_ae(self, year):
        """
        Load the AE index from the year, from the config.AE_DIR directory.
        It looks for a AE file with the "YYYY*ae.txt" file format and expects
        two columns: a datetime column and an AE column.

        Parameters
        ----------
        year: str or int
            The year to load the data from.
        
        Returns
        -------
        ae_data: pd.DataFrame
            A pd.DataFrame object containing the datetime index and AE column.
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
        Saves the merged catalog to a csv file with the same name as 
        the loaded catalog.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.catalog.to_csv(self.catalog_path, index_label='dateTime')

if __name__ == "__main__":
    m = Merge_AE(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_catalog_04.csv' ))
    m.loop()
    m.save_catalog()