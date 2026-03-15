# import matplotlib.gridspec as gridspec
# import matplotlib.pyplot as plt
# import seaborn as sns
# import yaml
from astroos_pipelines.utils.plots.dataset_eda import dataset_eda

from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

import sys
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from astropy import units as u
import pandas as pd
import importlib

from astroos_pipelines.lsst.query import AstroosQueryLSST
from astroos_pipelines.pipelines import StagePipeline 
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])


from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


# ============================================================
# StageHSTCatalogQuery
# ============================================================
class StageHSTCatalogQuery(StagePipeline):
    """
    Data pipeline stage for querying HST catalog.
    """
    def __init__(self):
        super().__init__(stage_name="hst_catalog_query", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return True

    def run(self):

        fits_file = ""

        # load arbitrary fits for eda using astropy
        # file = "/Users/davidsergio/thesis-org/hst/hst.fits"
        file = "catalogs/hst/hst.fits"
        # hdul = fits.open(file)
        # print(f"Number of HDUs in the FITS file: {len(hdul)}")
        # hdu = hdul[1]
        # print(f"HDU name: {hdu.name}")

        table = Table.read(file, hdu=1)

        # only first max_records rows for eda
        max_records = self.pipeline.max_records 

        if len(table) > max_records:
            log.warning(f"Table has {len(table)} records, but only using first {max_records} for EDA.")
            table = table[:max_records]

        self.output = table


# ============================================================
# StageHSTExploratoryDataAnalysis
# ============================================================
class StageHSTExploratoryDataAnalysis(StagePipeline):
    """
    Data pipeline stage for exploratory data analysis of HST catalog.
    """
    def __init__(self):
        super().__init__(stage_name="eda", requires_stage_dir=True)

    def _validate_prev_stage(self):
        return True

    def run(self):

        # read the table from the previous stage 
        table = self.prev_stage.output

        # print("Basic statistics of the catalog:")
        # print(table.info)

        # plot distributions of key features

        # df = table.to_pandas()
        # columns = df.columns

        include_columns = {  
                           'ra': "Right Ascension",
                           'dec': "Declination",
                           'jh_mag': "J-H color",
                           'z_spec': "Spectroscopic redshift",
                           'z_peak_phot': "Photometric redshift (peak)",
                           'z_peak_grism': "Grism redshift (peak)",
                           'z_best': "Best redshift",
                           'sfr': "Star Formation Rate",
                           'sfr_IR': "Star Formation Rate from IR",
                           'sfr_UV': "Star Formation Rate from UV",
                           'lmass': "Log Stellar Mass",
                           'Av': "Visual Extinction",
                           'beta': "UV slope",
                           'L_IR': "Infrared Luminosity",
                           'chi2': "Chi-squared of SED fit",
            } 
        dataset_eda(table=table, columns=include_columns, save_dir=self.stage_dir, title="HST")

        # save all columns to a yaml file for reference
        # also include description of the columns if available
        # with open(f"{self.stage_dir}/columns.yaml", "w") as f:
            # yaml.dump(columns.tolist(), f)
        
        # print table info to file
        # with open(f"{self.stage_dir}/table_info.txt", "w") as f:
            # f.write(str(table.info))
                
        # include_columns = [  
            # 'ra',
            # 'dec',
            # 'jh_mag',
            # 'z_spec',
            # 'z_peak_phot',
            # 'z_peak_grism',
            # 'z_best',
            # 'sfr',
            # 'sfr_IR',
            # 'sfr_UV',
            # 'lmass',
            # 'Av',
            # 'beta',
            # 'L_IR',
            # 'chi2',
         # ]

        # columns = [col for col in columns if col in include_columns]

        # # print the columns being plotted
        # print(f"Plotting distributions for columns: {columns}")

        # n_cols = 3  # number of subplot columns
        # n_rows = int(np.ceil(len(columns) / n_cols))

        # fig = plt.figure(constrained_layout=True, figsize=(5 * n_cols, 4 * n_rows))
        # fig.suptitle("HST EDA", fontsize=24)

        # gs = gridspec.GridSpec(
            # n_rows,
            # n_cols,
            # figure=fig,
            # width_ratios=[1.0] * n_cols,
            # height_ratios=[1.0] * n_rows,
        # )

        # for i, col in enumerate(columns):
            # r = i // n_cols
            # c = i % n_cols

            # ax = fig.add_subplot(gs[r, c])

            # data = df[col].dropna()

            # # drop inf values
            # data = data.replace([np.inf, -np.inf], np.nan).dropna()

            # sns.histplot(data, bins=50, kde=True, ax=ax)

            # ax.set_title(f"Distribution of {col}")
            # ax.set_xlabel(col)
            # ax.set_ylabel("Count")

        # # remove empty axes if grid > number of columns
        # for j in range(len(columns), n_rows * n_cols):
            # fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

        # plt.savefig(f"{self.stage_dir}/eda_histograms.png")
        # print(f"Saved EDA plot to {self.stage_dir}/eda_histograms.png")
        # plt.close()


