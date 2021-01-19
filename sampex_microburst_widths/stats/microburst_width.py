# Histogram the microburst widths
import pathlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sampex_microburst_widths import config
from sampex_microburst_widths.misc import plot_annotator_decorator

# Catalog 0 uses A1000
# Catalog 2 uses A500
# Catalog 3 uses A2000

catalog_params = {
    'microburst_catalog_00.csv':1000,
    'microburst_catalog_02.csv':500,
    # 'microburst_catalog_03.csv':2000,
}

@plot_annotator_decorator.annotate_plot
def main():
    fig, ax = plt.subplots()

    for cat_name, background_width in catalog_params.items():
        df = pd.read_csv(pathlib.Path(config.PROJECT_DIR, 'data', cat_name))

        ax.hist(df['width_s'], bins=np.linspace(0, 0.5, num=20), histtype='step', 
                label=r'$A_{{{0}}}$'.format(background_width))
        ax.set_title('SAMPEX-HILT | >1 MeV Microburst Widths')
        # ax.set_yscale('log')
        ax.set_xlabel('Width [s]')
        # ax.text(0.95, 0.95, "O'Brien burst parameter\nWidth at half prominence\nno visual inspection", 
        #         ha='right', va='top', transform=ax.transAxes
        #         )

    plt.legend()
main()
plt.show()