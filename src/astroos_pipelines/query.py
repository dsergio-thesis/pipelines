
import sys
from astropy.table import Table
from abc import ABC, abstractmethod

import sys
import importlib

# from astroos_pipelines.logger.logger import setup_logging
# importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
# import logging
# setup_logging()
# log = logging.getLogger(__name__)


class AstroosQuery(ABC):
    """
    Abstract Base Class for AstroosQuery
    """

    def __init__(self, 
                 res_object_identifier_column, 
                 root_dir
                 ):
        """
        Initialize AstroosQuery with a root directory and timeout.

        Parameters
        ----------
        root_dir : str, optional
            The root directory for saving data (default is current working directory).
        """

        self.root_dir = root_dir
        self.res_table = Table()
        self.res_object_identifier_column = res_object_identifier_column
        
