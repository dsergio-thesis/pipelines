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
from astroos_pipelines.dag import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.pipelines'])
importlib.reload(sys.modules['astroos_pipelines.dag'])

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class HSTNodeCatalog(Node):
    def __init__(self,
             node_type="catalog_hst",
             node_id=None,
             parents=[],
             parameters=None,
             inputs=[],
             outputs=[]):
        super().__init__(node_type=node_type,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "HSTNodeCatalog"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):
        file = "catalogs/hst/hst.fits"

        table = Table.read(file, hdu=1)

        # only first max_records rows for eda
        max_records = self.parameters['max_records'] 

        if len(table) > max_records:
            log.warning(f"Table has {len(table)} records, but only using first {max_records} for EDA.")
            table = table[:max_records]

        self.output_fits_table(table)

        

class HSTNodeEDA(Node):
    def __init__(self,
             node_type="catalog_hst_eda",
             node_id=None,
             parents=[],
             parameters=None,
             inputs=[],
             outputs=[]):
        super().__init__(node_type=node_type,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "HSTNodeEDA"
        return d

    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        # read fits table from input artifact
        print(f"running EDA {self.inputs}")
        artifact = self.inputs[0]
        table = Table.read(artifact.file_path, hdu=1)

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
        dataset_eda(table=table, 
                    columns=include_columns, 
                    save_dir=f"_pipelines/{self.node_id}", 
                    title="HST")


