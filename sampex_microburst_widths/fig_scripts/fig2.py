"""
This script generates a histogram of all microburst widths on the left panel, and
the widths as a function of AE in the right panel.

Parameters
----------
catalog_name: str
    The name of the catalog in the config.PROJECT_DIR/data/ directory.
r2_thresh: float
    The adjusted R^2 threshold for the fits. I chose a default value of 0.9.
max_width: float
    Maximum microburst width (FWHM) in seconds to histogram. A good default is
    0.25 [seconds]
"""
import pathlib
import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.fig_scripts import fig_params


plt.rcParams.update({'font.size': 13})

def calc_percentiles(df, percentiles):
    return df[width_key].quantile(q=percentiles/100)

def plot_full_dist(ax, df, width_bins, width_percentiles):
    """

    """
    ax.hist(df[width_key], bins=width_bins, color='k', histtype='step', density=True)
    s = (
        f"Percentiles [ms]"
        f"\n25%: {(width_percentiles.loc[0.25]).round().astype(int)}"
        f"\n50%: {(width_percentiles.loc[0.50]).round().astype(int)}"
        f"\n75%: {(width_percentiles.loc[0.75]).round().astype(int)}"
    )
    ax.text(0.64, 0.9, s, 
            ha='left', va='top', transform=ax.transAxes
            )
    plt.suptitle('Distribution of > 1 MeV Microburst Duration\nSAMPEX/HILT')
    # Left panel tweaks
    ax.set_xlim(0, params['max_width_ms'])
    ax.set_ylabel('Probability Density')
    ax.set_xlabel('FWHM [ms]')
    ax.text(0, 0.99, f'(a) All Microbursts', va='top', transform=ax.transAxes, 
        weight='bold', fontsize=15)
    return

def plot_ae_dist(ax, df, ae_bins):
    """

    """
    median
    # 1 AE < ae_bins[0]
    df_flt = df[df['AE'] < ae_bins[0]]
    ax.hist(df_flt[width_key], bins=width_bins, histtype='step', density=True,
        label=f'AE [nT] < {ae_bins[0]}', lw=2)
    print(f'Median microburst width for AE [nT] < {ae_bins[0]} is '
            f'{round(df_flt[width_key].median())} ms | N = {df_flt.shape[0]}')

    # 2. Bracketed AE. 
    for start_ae, end_ae in zip(ae_bins[:-1], ae_bins[1:]):
        df_flt = df[(df['AE'] > start_ae) & (df['AE'] < end_ae)]

        ax.hist(df_flt[width_key], bins=width_bins, histtype='step', density=True,
                label=f'{start_ae} < AE [nT] < {end_ae}', lw=2)
        print(f'Median microburst width for {start_ae} < AE [nT] < {end_ae} is '
            f'{round(df_flt[width_key].median())} ms | N = {df_flt.shape[0]}')

    # 3. AE > ae_bins[-1]
    df_flt = df[df['AE'] > ae_bins[-1]]
    ax.hist(df_flt[width_key], bins=width_bins, histtype='step', density=True,
        label=f'AE [nT] > {ae_bins[-1]}', lw=2)
    print(f'Median microburst width for AE [nT] > {ae_bins[-1]} is '
            f'{round(df_flt[width_key].median())} ms | N = {df_flt.shape[0]}')

    ax.legend(loc='center right', fontsize=12)
    ax.set_xlabel('FWHM [ms]')

    ax.text(0, 0.99, f'(b) As a function of AE', va='top', transform=ax.transAxes, 
        weight='bold', fontsize=15)
    return

def make_figure(percentiles, ae_bins):
    """
    The script, just wrapped in a function.
    """
    ### Load the data ###
    df = fig_params.load_catalog()
    width_percentiles = calc_percentiles(df, percentiles)

    ### Plot ###
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(10, 5))

    # Left panel histogram and statistics.
    plot_full_dist(ax[0], df, width_bins, width_percentiles)

    # Right Panel 
    plot_ae_dist(ax[1], df, ae_bins)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    ### Script parameters ###
    params = fig_params.params
    width_key = params['width_key']

    width_bins = np.linspace(0, params['max_width_ms']+0.001, num=50)
    ae_bins = [100, 300]
    percentiles = np.array([25, 50, 75])

    make_figure(percentiles, ae_bins)