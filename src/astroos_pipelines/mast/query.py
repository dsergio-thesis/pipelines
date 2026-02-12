
import sys
from astropy.table import Table

from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroos_pipelines.query import AstroosQuery

import sys
import importlib
from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class AstroosQueryMast(AstroosQuery):
    """
    MAST query class for querying astronomical data from the MAST archive.

    Parameters
    ----------
    root_dir : str, optional
        The root directory for saving data (default is current working directory).
    """

    def __init__(self, 
                 res_object_identifier_column, 
                 root_dir
                 ):

        self.root_dir = root_dir
        self.res_table = Table()
        self.res_object_identifier_column = res_object_identifier_column
        
    def query(self, query_params):
        """
        Perform the MAST query using the provided query parameters.

        Parameters
        ----------
        query_params : dict
            A dictionary containing the query parameters for the MAST archive.

        Returns
        -------
        Table
            An astropy Table containing the results of the MAST query.
        """
        log.info("Performing MAST query with parameters: %s", query_params)

        # Perform the MAST query using astroquery
        coord = SkyCoord(ra=query_params['ra'], dec=query_params['dec'], unit=(u.deg, u.deg))
        obs = Observations.query_region(coord, radius=0.01*u.deg)
        hst = obs[obs["obs_collection"] == "HST"]
        self.res_table = Table(hst)
        
        
        log.info("MAST query completed. Number of results: %d", len(self.res_table))
        
        return self.res_table
