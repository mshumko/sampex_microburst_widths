"""
This module makes sure that all of the figure scripts use the same 
microburst catalog parameters, and it contains the load_catalog 
function.
"""
import pathlib

import pandas as pd

import sampex_microburst_widths.config as config


### Parameters for Figs. 2-4
params = {
    'catalog_name':'microburst_catalog_04.csv',
    'max_width_ms':500,
    'min_r2':0.9,
    'width_key':'fwhm_ms'
}


def load_catalog(catalog_name=None, max_width_ms=None, min_r2=None):
    """
    Load the microburst catalog and manipulate it for plotting.

    Manipulations:
    1. Drop NaN values that are typically for the SAMPEX attitude,
    2. Scale the width_s and fwhm durations to milliseconds,
    3. Filter out microbursts wider than max_width_ms,
    4. Change fwhm_ms to abs(fwhm_ms) (+/- sigma is redundant for a Gaussian)
    5. Apply the min_r2 filter.
    
    Parameters
    ----------
    catalog_name: str
    
    max_width_ms: int
    
    min_r2: float

    Returns
    -------
    """
    # If the user does not specify inputs, use the global variables
    # defined above.
    if catalog_name is None:
        catalog_name = params['catalog_name']
    if max_width_ms is None:
        max_width_ms = params['max_width_ms']
    if min_r2 is None:
        min_r2 = params['min_r2']

    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', catalog_name))
    df.dropna(inplace=True)
    df['width_ms'] = 1000*df['width_s'] # Convert seconds to ms.
    df['fwhm_ms'] = 1000*df['fwhm']
    df = df[df['width_ms'] < max_width_ms]
    df['fwhm_ms'] = df['fwhm_ms'].abs()
    df = df[df.adj_r2 > min_r2]
    return df