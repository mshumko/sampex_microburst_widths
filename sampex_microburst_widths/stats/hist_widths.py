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

    plt.hist(df['width_s'], bins=np.linspace(0, 1, num=50), color='k', histtype='step')
    plt.title('SAMPEX-HILT | >1 MeV Microburst Widths')
    plt.yscale('log')
    plt.xlabel('Width [s]')
    plt.text(0.95, 5E4, "O'Brien burst parameter\nWidth at half prominence\nno visual inspection", ha='right', va='top')

main()
plt.show()