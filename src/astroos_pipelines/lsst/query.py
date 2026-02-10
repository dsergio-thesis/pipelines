
import sys
from astropy.table import Table
import importlib

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    from lsst.rsp.utils import get_pyvo_auth
    import lsst.geom as geom

    # other LSST dependencies
    from pyvo.dal.adhoc import DatalinkResults
    from astropy.time import Time
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery

    rsp_mode = True
except ImportError:
    pass

from astroos_pipelines.query import AstroosQuery
importlib.reload(sys.modules['astroos_pipelines.query'])

import sys
import importlib
from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class AstroosQueryLSST(AstroosQuery):
    """
    AstroosQuery LSST Client
    """

    def __init__(self, root_dir=None, credentials_file=None, max_records=None):
        super().__init__(
            res_object_identifier_column="ObjectId",
            root_dir=root_dir,
        )

        self.credentials_file=credentials_file

        # self.tap_service = PyvoTAPClient(base_url="https://data.lsst.cloud/api/tap", maxrecords=max_records)
                                        #  credentials_file=self.credentials_file,

        self.tap_service = get_tap_service("tap")                

        # result = self.tap_service.sync("select top 10 objectId from dp1.Object")
        # print("LSST TAP Query Result:")
        # print(result)

        log.info("Initialized LSST Query Client.")

    def __repr__(self):
        return f"<AstroosQueryLSST(root_dir={self.root_dir}, timeout={self.timeout})>"
    
    def query(self, query):
        """
        Query LSST using ADQL.

        Parameters
        ----------
        query : str
            The ADQL query string.

        Returns
        -------
        result : table astropy.table.Table
            The query result
        """
        log.debug(f"Querying LSST ADQL with query:\n{query}")

        res = self.tap_service.search(query)
        table = Table(res.to_table())

        if res is None:
            log.debug("No results found.")
            return Table()
        
        log.debug("LSST Query Result:")
        log.debug(res)

        return table

    def query_async(self, query):
        """
        Query LSST using ADQL asynchronously.

        Parameters
        ----------
        query : str
            The ADQL query string.

        Returns
        -------
        result : table
            The query result
        """
        log.info(f"Async Querying LSST ADQL with query:\n{query}")

        job = self.tap_service.async_submit(query)
        self.tap_service.async_wait(job)
        res = self.tap_service.async_result(job)
        log.debug(f"Async LSST Query Result:")
        log.debug(res)
        table = Table(res.to_table())

        return table

