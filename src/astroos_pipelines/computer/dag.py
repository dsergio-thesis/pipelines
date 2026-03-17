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
from astroos_pipelines.dag import *
from astroos_pipelines.computer.datases import *
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset

importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.dag'])
importlib.reload(sys.modules['astroos_pipelines.computer.datases'])

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class ComputerNodeCatalog(Node):
    def __init__(self,
                 node_type="catalog_computer",
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
        d["type"] = "ComputerNodeCatalog"
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
        pass
