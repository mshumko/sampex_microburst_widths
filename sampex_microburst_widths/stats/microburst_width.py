# Histogram the microburst widths
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

@plot_annotator_decorator.annotate_plot
def main():
    df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', 'microburst_test_catalog.csv'))

    fig, ax = plt.subplots()
    ax.hist(df['width_s'], bins=np.linspace(0, 0.2, num=20), color='k', histtype='step')
    ax.set_title('SAMPEX-HILT | >1 MeV Microburst Widths')
    ax.set_yscale('log')
    ax.set_xlabel('Width [s]')
    ax.text(0.95, 0.95, "O'Brien burst parameter\nWidth at half prominence\nno visual inspection", 
            ha='right', va='top', transform=ax.transAxes
            )

main()
plt.show()