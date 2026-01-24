import os
import sys
import re
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import requests
from io import StringIO
import pandas as pd
from PIL import Image

# astroquery
from astroquery.ipac.ned import Ned
from astroquery.simbad import Simbad
from astroquery.sdss import SDSS
from astroquery.fermi import FermiLAT
from astroquery.skyview import SkyView


if (sys.modules.get('src.astroos.utils.rsp_utils') is not None): 
    del sys.modules['src.astroos.utils.rsp_utils']

from src.astroos.utils.rsp_utils import init_rsp_mode, rsp_mode
init_rsp_mode()

# other LSST dependencies
from pyvo.dal.adhoc import DatalinkResults

# astropy
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table
from astropy.visualization import (ImageNormalize, PercentileInterval, AsinhStretch)
from astropy.io import fits

if (sys.modules.get('src.astroos.morphometryka') is not None): 
    del sys.modules['src.astroos.morphometryka']
from src.astroos.morphometryka import simple_segmentation, measure_morfometry, morfomytry
from abc import ABC, abstractmethod

if (sys.modules.get('src.astroos.tap_clients') is not None): 
    del sys.modules['src.astroos.tap_clients']
from src.astroos.tap_clients import PyvoTAPClient

# ============================================================
# AstroosQuery Abstract Base Class
# ============================================================
class AstroosQuery(ABC):
    """
    Abstract Base Class for AstroosQuery
    """

    def __init__(self, 
                 res_object_identifier_column, 
                 root_dir, 
                 timeout=120):
        """
        Initialize AstroosQuery with a root directory and timeout.
        Parameters
        ----------
        root_dir : str, optional
            The root directory for saving data (default is current working directory).
        timeout : int, optional
            Timeout for queries in seconds (default is 120 seconds).
        """

        self.root_dir = root_dir

        self.timeout = timeout

        
        self.res_table = Table()
        self.res_object_identifier_column = res_object_identifier_column

    def __len__(self):
        return len(self.res_table)

    def __getitem__(self, key):
        return self.res_table[key]

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError

    def cache_astropy_table(self, table: Table, filename: str):
        """
        Cache an Astropy Table to a file.

        Parameters
        ----------
        table : astropy.table.Table
            The Astropy Table to cache.
        filename : str
            The filename to save the table to.
        """
        filepath = os.path.join(self.root_dir, "astroquery_cache", filename)
        table.write(filepath, format='ascii.csv', overwrite=True)
        print(f"Cached table to {filepath}")
    
    def get_result_table_from_cache(self, filename: str):
        """
        Retrieve an Astropy Table from a cached file.

        Parameters
        ----------
        filename : str
            The filename to load the table from.

        Returns
        -------
        table : astropy.table.Table
            The loaded Astropy Table. If the file does not exist, returns None.
        """
        filepath = os.path.join(self.root_dir, "astroquery_cache", filename)
        if os.path.exists(filepath):
            table = Table.read(filepath, format='ascii.csv')
            print(f"Loaded cached table from {filepath}")
            return table
        else:
            return None
    
# ============================================================
# AstroosAstroqueryClient Abstract Base Class
# ============================================================
class AstroosAstroqueryClient(AstroosQuery):
    """
    Astroquery Client Base Class
    """

    def __init__(self,
                 astroquery_api,
                 astroquery_api_label,
                 res_object_identifier_column,
                 root_dir,
                 timeout=120):
        super().__init__(
            res_object_identifier_column=res_object_identifier_column,
            root_dir=root_dir,
            timeout=timeout
        )
        self.astroquery_api = astroquery_api
        self.astroquery_api_label = astroquery_api_label

    def query_region(self, coordinates, radius, limit=None):
        """
        Query a region of the sky given coordinates and radius.

        Parameters
        ----------
        api : Astroquery API class
            The Astroquery API class to use (e.g., Ned, Simbad).
        api_label : str
            A label for the API (e.g., "NED", "Simbad").
        coordinates : astropy.coordinates.SkyCoord
            The coordinates of the region to query.
        radius : float
            The radius of the region to query, in arcseconds.

        Returns
        -------
        result : astropy.table.Table
            The query result as an Astropy Table.
        """

        # table filename for caching
        res_table_csv = f"query_result_{self.astroquery_api_label}" \
            f"_ra_{coordinates.ra.deg}_dec_{coordinates.dec.deg}" \
            f"_radius_{radius.to(units.arcsec).value}" \
            f"_limit_{limit if limit is not None else 'all'}.csv"

        # first check cache
        self.res_table = self.get_result_table_from_cache(res_table_csv)
        if self.res_table is not None:
            print(f"Loaded cached table from {res_table_csv}")
        else:
            
            self.res_table = self.astroquery_api.query_region(coordinates=coordinates, radius=radius)

            if self.res_table is None:
                print("No results found.")
                return Table()

            if limit is not None and len(self.res_table) > limit and len(self.res_table) > 0:
                self.res_table = self.res_table[:limit]
            self.cache_astropy_table(self.res_table, res_table_csv)

        return self.res_table
    
    def get_region_skyview_images(self, survey):
        """
        Get SkyView images for the query results.
        ** NOTE: ** Not useful except for thumbnail images.

        Parameters
        ----------
        survey : list or str
            The survey(s) to fetch images from (e.g., 'SDSSg', ['SDSSg', 'Fermi 5']).

        Returns
        -------
        images : list
            A list of fetched images. Does not include labels.
        """
        images = []
        for row in self.res_table:
            obj_name = row[self.res_object_identifier_column]
            print(f"Fetching images for {obj_name} from survey {survey}...")
            if survey is not None:
                fetched_images = SkyView.get_images(position=obj_name, survey=survey)
            else:
                fetched_images = SkyView.get_images(position=obj_name)
            images.extend(fetched_images)
        
        return images

# ============================================================
# AstroosQuery LSST
# ============================================================
class AstroosQueryLSST(AstroosQuery):
    """Astroquery LSST Client"""

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

        print("Initialized LSST Query Client.")

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
        print(f"Querying LSST ADQL with query:\n{query}")

        res = self.tap_service.search(query)
        table = Table(res.to_table())

        if res is None:
            print("No results found.")
            return Table()
        
        print("LSST Query Result:")
        print(res)

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
        print(f"Async Querying LSST ADQL with query:\n{query}")

        job = self.tap_service.async_submit(query)
        self.tap_service.async_wait(job)
        res = self.tap_service.async_result(job)
        print(f"Async LSST Query Result:")
        print(res)
        table = Table(res.to_table())

        return table



# ============================================================
# AstroosQuery NED
# ============================================================
class AstroosQueryNED(AstroosAstroqueryClient):
    """Astroquery NED Client"""

    def __init__(self, root_dir=None):
        super().__init__(
            astroquery_api=Ned, 
            astroquery_api_label="NED", 
            res_object_identifier_column="Object Name",
            root_dir=root_dir
        )
        print("Initialized NED Client")
        Ned.TIMEOUT = self.timeout

    def __repr__(self):
        return f"<AstroQueryNEDClient(root_dir={self.root_dir}, timeout={self.timeout})>"


# ============================================================
# AstroosQuery Simbad
# ============================================================
class AstroosQuerySimbad(AstroosAstroqueryClient):
    """Astroquery Simbad Client"""

    def __init__(self, root_dir=None):
        super().__init__(
            astroquery_api=Simbad, 
            astroquery_api_label="Simbad", 
            res_object_identifier_column="main_id",
            root_dir=root_dir
        )
        print("Initialized Simbad Client")
        Simbad.timeout = self.timeout

    def __repr__(self):
        return f"<AstroQuerySimbadClient(root_dir={self.root_dir}, timeout={self.timeout})>"


# ============================================================
# AstroosQuery SDSS
# ============================================================
class AstroosQuerySDSS(AstroosAstroqueryClient):
    """
    Astroquery SDSS Client
    See https://skyserver.sdss.org/dr16/en/help/docs/api.aspx#advanced for details.

    """

    def __init__(self, root_dir):
        super().__init__(
            astroquery_api=SDSS, 
            astroquery_api_label="SDSS", 
            res_object_identifier_column="main_id",
            root_dir=root_dir
        )
        self.sdss_base_url = "https://skyserver.sdss.org/dr17/SkyServerWS/SearchTools/SqlSearch"

        print("Initialized SDSS Client")
        SDSS.timeout = self.timeout

    def __repr__(self):
        return f"<AstroQuerySDSSClient(root_dir={self.root_dir}, timeout={self.timeout})>"
    
    def query_adql(self, query, format='csv'):
        """
        Query SDSS using ADQL.

        Parameters
        ----------
        query : str
            The ADQL query string.
        format : str
            The format of the result ('csv', etc.).

        Returns
        -------
        result : astropy.table.Table
            The query result as an Astropy Table.
        """
        # print(f"Querying SDSS ADQL with query:\n{query}")
        request_url = \
            f"{self.sdss_base_url}?cmd={requests.utils.quote(query)}&format={format}"

        print(f"Request URL: {request_url}")

        res = None
        # res = requests.get(request_url)
        res = requests.get(request_url, timeout=self.timeout)
                           
        if res is None:
            print("No results found.")
            return Table()
        
        print(res.status_code)
        print(res.text)
        res = Table.read(StringIO(res.text), format=format)
        return res
    
    def query_TAP(self, query):
        """
        Query SDSS using TAP (Table Access Protocol).

        Parameters
        ----------
        query : str
            The ADQL query string.

        Returns
        -------
        result : astropy.table.Table
            The query result as an Astropy Table.
        """
        # print(f"Querying SDSS TAP with query:\n{query}")

        # res = SDSS.query_sql(query)
        res = Simbad.query_tap(query)

        if res is None:
            print("No results found.")
            return Table()

        return res
    
    def get_sdss_morph_dict(self):
        """
        Get SDSS morphology dictionary.

        Returns
        -------
        morph_dict : dict
            Dictionary mapping morphology codes to descriptions.
        """

        # check cache first
        morph_dict_file = os.path.join(self.root_dir, "astroquery_cache", "sdss_morph_dict.txt")
        if os.path.exists(morph_dict_file):
            with open(morph_dict_file, "r") as f:
                morph_dict = {}
                for line in f:
                    morph_type, object_count = line.strip().split(": ")
                    morph_dict[morph_type] = int(object_count)
            print(f"Loaded SDSS Morphology Dictionary from cache: {morph_dict}")
            return morph_dict
        else:
            res = self.query_TAP(
                """
                SELECT 
                b.morph_type,
                COUNT(*) AS object_count
                FROM basic AS b
                WHERE b.otype = 'G'
                AND b.morph_type IS NOT NULL
                GROUP BY b.morph_type
                ORDER BY object_count DESC

                ;

                """
            )
            morph_dict = {}
            for row in res:
                morph_dict[row['morph_type']] = row['object_count']

            print(f"SDSS Morphology Dictionary: {morph_dict}")
            # save dict to file
            morph_dict_file = os.path.join(self.root_dir, "astroquery_cache", "sdss_morph_dict.txt")
            with open(morph_dict_file, "w") as f:
                for morph_type, object_count in morph_dict.items():
                    f.write(f"{morph_type}: {object_count}\n")

            return morph_dict
    
    def get_sdss_objects_count(self):

        res = self.query_TAP(
            """
            -- get count of SDSS objects

            SELECT COUNT(*)
            FROM basic
            WHERE main_id LIKE 'SDSS%'

            ;

            """
        )
        # print(f"Total SDSS objects: {res}")
        return res

    def scan_TAP(self, 
                 ra_min=180, 
                 ra_max=190, 
                 dec_min=-90, 
                 dec_max=90, 
                 ra_offset=1, 
                 otype_longname_filter=None, 
                 limit=None):
        """
        Scan SDSS TAP service for everything in Dec +/-90 degrees,
        and for each range of RA values.

        Parameters
        ----------
        ra_min : float
            Minimum RA value.
        ra_max : float
            Maximum RA value.
        ra_offset : float
            RA offset for each query.

        Returns
        -------
        None
        """ 

        # n_queries = (ra_max - ra_min) / ra_offset
        # total_sdss = self.get_sdss_objects_count()['COUNT_ALL'][0]
        # n_records_per_query = total_sdss / n_queries

        # print(f'Total SDSS objects: {total_sdss}')
        # print(f"{n_records_per_query} records per query")

        query = \
        """
        -- Query all SDSS objects
        SELECT {limit} * 
        FROM basic b JOIN otypedef o ON b.otype = o.otype 
        WHERE main_id LIKE 'SDSS%' AND

        ra >= {ra_offset} AND
        ra < {ra_offset_limit} AND

        dec >= -90 AND
        dec <= 90 AND
        (o.otype_longname = 'Galaxy') AND
        (b.morph_type IS NOT NULL) AND
        1=1
        ;
        """

        queries = []

        # for i in range(ra_min, ra_max, ra_offset):
        #     # print(f'Query ra {i} to {i + ra_offset}')
        #     query_formatted = query.format(
        #         ra_offset=i,
        #         ra_offset_limit=i + ra_offset,
        #         dec_min=dec_min,
        #         dec_max=dec_max,
        #         limit = f"TOP {limit}" if limit is not None else "")
        #     queries.append((i, i + ra_offset, query_formatted))

        delta_ra = (ra_max - ra_min) % 360
        if np.isclose(delta_ra, 0.0) and not np.isclose(ra_min, ra_max):
            ra_vals = np.arange(0.0, 360.0, ra_offset)
        elif ra_min < ra_max:
            ra_vals = np.arange(ra_min, ra_max, ra_offset)
        else:
            # wrap case: region crosses RA=0
            ra_vals = np.concatenate([
                np.arange(ra_min, 360.0, ra_offset),
                np.arange(0.0, ra_max, ra_offset)
            ])

        for i in ra_vals:
            ra_lo = i
            ra_hi = (i + ra_offset) % 360
            query_formatted = query.format(
                ra_offset=ra_lo,
                ra_offset_limit=ra_hi,
                dec_min=dec_min,
                dec_max=dec_max,
                limit=f"TOP {limit}" if limit is not None else ""
            )
            queries.append((ra_lo, ra_hi, query_formatted))

        print(f'Total queries: {len(queries)}')
        # print(queries)

        query_index = 0
        max_queries = len(queries)
        # max_queries = 3

        total_records = 0

        for i in range(len(queries)):

            query_info = f"simbad_tap_sdss_{dec_min}_{dec_max}__{queries[query_index][0]}_{queries[query_index][1]}_limit{limit}"

            # first check cache
            if os.path.exists(f"{self.root_dir}/{query_info}.csv"):
                print(f"File {self.root_dir}/{query_info}.csv already exists. Skipping...")
                query_index += 1
                if query_index >= max_queries:
                    break
                continue

            try:

                """ 

                First SIMBAD to get overall catalog info


                """

                res = Simbad.query_tap(queries[query_index][2])

                """

                Then SDSS to get instrument specific things, such as run, camcol, field, id

                For this step, we use cross_id_async 

                """

                run_list = []
                camcol_list = []
                field_list = []
                id_list = []

                for row in res:
                    sdss_data = SDSS.query_crossid_async(
                        SkyCoord(row['ra'], row['dec'], unit=(units.deg, units.deg)), radius=5 * units.arcsec)
                    
                    # print()
                    # print(sdss_data.text)

                    if len(sdss_data.text.split("\n")) > 2:

                        header = sdss_data.text.split("\n")[1]
                        data = sdss_data.text.split("\n")[2]

                        if str(data).strip() == "":
                            run_list.append(None)
                            camcol_list.append(None)
                            field_list.append(None)
                            id_list.append(None)
                            continue

                        row_data = header + "\n" + data
                        df = pd.read_csv(StringIO(row_data))
                        print(f"objID: {df['objID']}")
                        objID = df['objID'].iloc[0]

                        decoded = decode_sdss_objid(objID)

                        run_list.append(decoded['run'])
                        camcol_list.append(decoded['camcol'])
                        field_list.append(decoded['field'])
                        id_list.append(decoded['id'])
    

                res['run'] = run_list
                res['camcol'] = camcol_list
                res['field'] = field_list
                res['id'] = id_list

                # print(res)
                print(f"Querying {queries[query_index][0]} to {queries[query_index][1]}: {len(res)} records found")

                total_records += len(res)

                df = res.to_pandas()
                df.to_csv(f"{self.root_dir}/{query_info}.csv", index=False)

            except Exception as e:
                print(f"Error querying {queries[query_index][0]} to {queries[query_index][1]}: {e}")
                raise e

            sleep(1.2)
            query_index += 1
            if query_index >= max_queries or total_records >= limit:
                break

    
    def scan(self, 
             pipeline,
             ra_min=180, 
             ra_max=190, 
             dec_min=-90, 
             dec_max=90, 
             ra_offset=1, 
             limit=None,
             ):
        """
        Scan

        Parameters
        ----------

        Returns
        -------
        Tuple: (positions, labels, metadata)
        """

        # testing
        dec_min = -90
        dec_max = 90
        ra_min = 150
        ra_max = 190
        ra_offset = (ra_max - ra_min) / 4
        test_set = [
            'SDSS J120024.69+233413.2',
            'SDSS J120314.70+233052.2',
            'SDSS J120224.56+292235.8',
            'SDSS J120302.39+381852.1',
            'SDSS J120341.95+611742.7',
            'SDSS J120322.59+625352.1',
            'SDSS J120129.26+683616.7',
            'SDSS J120425.45+041644.0',
            'SDSS J120628.48+633747.3',
            'SDSS J120920.78+261244.4',
            'SDSS J121111.99+390342.0',
            'SDSS J121314.09+152336.4'
        ]
        test_set = test_set[:limit] if limit is not None else test_set

        print(f"Scanning RA {ra_min} to {ra_max}, Dec {dec_min} to {dec_max}")

        query = \
        """
        SELECT {limit_str} * 
        FROM basic b JOIN otypedef o ON b.otype = o.otype 
        WHERE main_id LIKE 'SDSS%' AND

        ra >= {ra_offset} AND
        ra < {ra_offset_limit} AND
        dec >= {dec_min} AND
        dec <= {dec_max} AND
        (o.otype_longname = 'Galaxy') AND
        ((b.main_id IN ('{test_set}')) OR 1=1) AND
        b.rvz_redshift < 0.05 AND
        (b.morph_type IS NOT NULL)

        ;
        """


        query = \
        """
        SELECT {limit_str} *
        FROM basic b JOIN otypedef o ON b.otype = o.otype
        WHERE main_id LIKE 'SDSS%'
        AND b.rvz_redshift BETWEEN 0.0 AND 0.05
        AND o.otype_longname = 'Galaxy'
        AND b.main_id like 'SDSS%'
        AND (b.morph_type IS NOT NULL)
        AND b.morph_type IN ('E', 'S0', 'Sa', 'Sb', 'Sc', 'Sd', 'Irr', 'SBa', 'SBb', 'SBc', 'SBd', 'cD', 'dE', 'dS0', 'dIrr')
        """

        test_set = [
            'LBQS 1313-0138',
            'FIRST J131531.2-013131',
            'ATO J199.0886-01.7543'
        ]
        query = \
        """
        SELECT {limit_str} * 
        FROM basic b JOIN otypedef o ON b.otype = o.otype 
        WHERE main_id LIKE 'SDSS%' AND
        ((b.main_id IN ('{test_set}')) OR 1=1)
        ;
        """
        queries = []

        delta_ra = (ra_max - ra_min) % 360
        if np.isclose(delta_ra, 0.0) and not np.isclose(ra_min, ra_max):
            ra_vals = np.arange(0.0, 360.0, ra_offset)
        elif ra_min < ra_max:
            ra_vals = np.arange(ra_min, ra_max, ra_offset)
        else:
            # wrap case: region crosses RA=0
            ra_vals = np.concatenate([
                np.arange(ra_min, 360.0, ra_offset),
                np.arange(0.0, ra_max, ra_offset)
            ])

        for i in ra_vals:
            ra_lo = i
            ra_hi = (i + ra_offset) % 360
            query_formatted = query.format(
                ra_offset=ra_lo,
                ra_offset_limit=ra_hi,
                dec_min=dec_min,
                dec_max=dec_max,
                test_set="','".join(test_set),
                limit_str=f"TOP {limit}" if limit is not None else ""
            )
            queries.append((ra_lo, ra_hi, query_formatted))
            print(f"query: {query_formatted}")

        print(f'Total queries: {len(queries)}')
        # print(queries)

        query_index = 0
        max_queries = len(queries)
        total_records = 0

        overall_df = pd.DataFrame()

        label_definitions = pipeline.dataset.labels

        for i in range(len(queries)):

            try:
                """ 
                First SIMBAD to get overall catalog info
                """

                res = Simbad.query_tap(queries[query_index][2])

                """
                Then SDSS to get instrument specific things, such as run, camcol, field, id
                For this step, we use cross_id_async 
                """

                # print(res)
                print(f"Querying {queries[query_index][0]} to {queries[query_index][1]}: {len(res)} records found")

                positions = []
                coords_ra = []
                coords_dec = []
                labels = []
                main_ids = []
                redshifts = []
                maj_axs = []
                min_axs = []
                angles = []
                band_image_components = {}
                bands = ['u', 'g', 'r', 'i', 'z']
                for band in bands:
                    band_image_components[f"{band}_rerun"] = []
                    band_image_components[f"{band}_run"] = []
                    band_image_components[f"{band}_run06d"] = []
                    band_image_components[f"{band}_camcol"] = []
                    band_image_components[f"{band}_field"] = []
                    band_image_components[f"{band}_field04d"] = []

                for row in res:
                    sdss_data = SDSS.query_crossid_async(
                        SkyCoord(row['ra'], row['dec'], unit=(units.deg, units.deg)), radius=5 * units.arcsec)
                    
                    # print()
                    # print(sdss_data.text)

                    if len(sdss_data.text.split("\n")) <= 2:
                        print(f"No SDSS data found for {row['main_id']}. Skipping...")
                        continue
                    else:

                        header = sdss_data.text.split("\n")[1]
                        data = sdss_data.text.split("\n")[2]

                        if str(data).strip() == "":
                            print(f"No SDSS data found for {row['main_id']}. Skipping...")
                            continue

                        morph_type = str(row['morph_type'])
                        
                        label_index = label_definitions._get_label_index(morph_type)

                        print(f"Found morph_type: {morph_type} for {row['main_id']} with label index {label_index}")

                        coords_ra.append(row['ra'])
                        coords_dec.append(row['dec'])
                        main_ids.append(row['main_id'])
                        redshifts.append(row['rvz_redshift'])
                        maj_axs.append(row['galdim_majaxis'])
                        min_axs.append(row['galdim_minaxis'])
                        angles.append(row['galdim_angle'])
                        positions.append((row['ra'], row['dec']))
                        labels.append(label_index)
                        

                        row_data = header + "\n" + data
                        df = pd.read_csv(StringIO(row_data))
                        # print(f"objID: {df['objID']}")
                        objID = df['objID'].iloc[0]
                        decoded = decode_sdss_objid(objID)
                        
                        rerun = decoded['rerun']
                        run = decoded['run']
                        run06d = f"{run:06d}"
                        camcol = decoded['camcol']
                        field = decoded['field']
                        field04d = f"{field:04d}"
                        id = decoded['id']

                        for band in bands:

                            # filename = "frame-{band}-{run:06d}-{camcol}-{field:04d}.fits.bz2"

                            band_image_components[f"{band}_rerun"].append(rerun)
                            band_image_components[f"{band}_run"].append(run)
                            band_image_components[f"{band}_run06d"].append(run06d)
                            band_image_components[f"{band}_camcol"].append(camcol)
                            band_image_components[f"{band}_field"].append(field)
                            band_image_components[f"{band}_field04d"].append(field04d)

                total_records += len(coords_ra)
                print(f"Accumulated total records: {total_records + len(coords_ra)}")
                
                df = pd.DataFrame({
                    "ra": coords_ra,
                    "dec": coords_dec,
                    "label": labels,
                    "main_id": main_ids,
                    "rvz_redshift": redshifts,
                    "galdim_majaxis": maj_axs,
                    "galdim_minaxis": min_axs,
                    "galdim_angle": angles,
                })
                
                for band in bands:
                    df[f"{band}_rerun"] = band_image_components[f"{band}_rerun"]
                    df[f"{band}_run"] = band_image_components[f"{band}_run"]
                    df[f"{band}_run06d"] = band_image_components[f"{band}_run06d"]
                    df[f"{band}_camcol"] = band_image_components[f"{band}_camcol"]
                    df[f"{band}_field"] = band_image_components[f"{band}_field"]
                    df[f"{band}_field04d"] = band_image_components[f"{band}_field04d"]
                
                overall_df = pd.concat([overall_df, df], ignore_index=True)

                image_url_format_string = "https://data.sdss.org/sas/dr17/eboss/photoObj/frames/" \
                            "{rerun}/{run}/{camcol}/frame-{band}-{run06d}-{camcol}-{field04d}.fits.bz2"

                # result = tuple(df[col].tolist() for col in df.columns)
                # result = result, image_url_format_string

            except Exception as e:
                print(f"Error querying {queries[query_index][0]} to {queries[query_index][1]}: {e}")
                raise e

            sleep(1.2)
            query_index += 1
            if query_index >= max_queries or total_records >= limit:
                break

        overall_df.to_csv(f"{self.root_dir}/labeled_catalog_with_image_components.csv", index=False)
        overall_df.to_csv(f"{self.root_dir}/positions_labels.csv", index=True)
        return overall_df, image_url_format_string
    














































def decode_sdss_objid(objid: int):
    """
    Decode an SDSS objID into its components:
    skyVersion, rerun, run, camcol, field, id
    """
    sky_version = (objid >> 60) & 0xF
    rerun       = (objid >> 48) & 0xFFF
    run         = (objid >> 32) & 0xFFFF
    camcol      = (objid >> 29) & 0x7
    field       = (objid >> 16) & 0x1FFF
    obj         = objid & 0xFFFF

    return {
        "objID": objid,
        "skyVersion": sky_version,
        "rerun": rerun,
        "run": run,
        "camcol": camcol,
        "field": field,
        "id": obj
    }































class AstroQueryUtils:
    """Utility class for Astroquery operations"""

    @staticmethod
    def get_image_from_file(fits_filename):
        return [fits.open(fits_filename)]

    @staticmethod
    def get_ra_dec_in_hms(ra_dec_hms):

        regex = r'([\-\+]?[0-9]+\s+[0-9]+\s+[0-9]+\.[0-9]+)'
        m = re.findall(regex, ra_dec_hms)
        print(f"m: {m}")
        if (len(m) == 2):
            ra_hms = m[0]
            dec_hms = m[1]
        return ra_hms, dec_hms

    @staticmethod
    def convert_ra_dec_hms_to_deg(ra_dec_hms):

        ra_hms, dec_hms = AstroqueryClient.get_ra_dec_in_hms(ra_dec_hms)

        # convert ra from hms to deg
        coordinates = SkyCoord(
            ra=ra_hms,
            dec=dec_hms,
            unit=(units.hourangle, units.deg),
            frame='icrs'
        )
        print(f"Converted to deg: RA: {coordinates.ra.deg}, DEC: {coordinates.dec.deg}")
        return coordinates.ra.deg, coordinates.dec.deg
    
    @staticmethod
    def plot_image(image: np.ndarray, name):
        data = np.squeeze(image)
        norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch())

        plt.figure(figsize=(8, 8))
        plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title(f"{name}")
        plt.savefig(f"{name}.png")
        plt.close()
    
    @staticmethod
    def validate_hdul(hdul: fits.HDUList, required_keywords: list = None):
        """
        Validates and fixes common issues in a FITS file.
        
        Parameters:
        - hdul: HDUList: The FITS file to validate.
        - required_keywords: list: List of required keywords to check for existence.
        
        Returns:
        - None: The function will modify the FITS file in place.
        """
        # Open the FITS file

        header = hdul[0].header

        # if ('OBS NAME' in hdul[0].header):
        #     hdul[0].header['OBS-NAME'] = hdul[0].header['OBS NAME']
        #     del hdul[0].header['OBS NAME']
        
        # Check for valid keyword names and fix them
        for key in list(header.keys()):
            # Check and correct the keyword name length
            if len(key) > 8:
                # print(f"Keyword '{key}' is too long. Replacing with '{key[:8]}'")
                value = header[key]
                header.remove(key)
                header[key[:8]] = value
            
            # Check if the keyword has illegal characters or spaces
            if any(char not in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_' for char in key):
                fixed_key = key.replace(' ', '_')[:8]  # Replace spaces with underscores and limit length
                # print(f"Keyword '{key}' is invalid. Renaming to '{fixed_key}'")
                
                if key in header:
                    value = header[key]
                    header.remove(key)
                    header[fixed_key] = value
        
        # Check for duplicate keywords and remove duplicates
        # seen_keys = set()
        # duplicates = []
        # for key in header.keys():
        #     if key in seen_keys:
        #         duplicates.append(key)
        #     else:
        #         seen_keys.add(key)
        
        # for dup in duplicates:
        #     print(f"Removing duplicate keyword '{dup}'")
        #     header.remove(dup)
        
        # Check for required keywords
        if required_keywords:
            for req_key in required_keywords:
                if req_key not in header:
                    # print(f"Required keyword '{req_key}' not found. Adding default value.")
                    # You can set a default value if necessary
                    header[req_key] = 'DEFAULT'
        
        # Check the value of the EXTEND keyword
        if 'EXTEND' in header:
            extend_value = header['EXTEND']
            
            if extend_value is None or extend_value not in [True, False, 'T', 'F']:
                # print(f"Invalid EXTEND value: {extend_value}. Setting EXTEND to True.")
                header['EXTEND'] = True  # Set to True if extensions exist
            
        else:
            # print("Adding EXTEND keyword with value True.")
            header['EXTEND'] = True  # Add the keyword if it doesn't exist
        
        # Check and fix problematic keywords
        for key in ['CD001001', 'CD002002', 'CALIBCON']: # just delete for now
            if key in header:
                value = header[key]
                print(f"Checking {key}: [{value}]")


                del header[key]

                # # Split the value to separate the numeric part from comments
                # if isinstance(value, str) and '/' in value:
                #     try:
                #         # Extract numeric value before the comment
                #         numeric_value = value.split()[0]  # Take the first part
                #         # Reassign the cleaned-up value without the comment
                #         header[key] = float(numeric_value)
                #         print(f"Fixed {key}: {header[key]}")
                #     except ValueError:
                #         print(f"Could not convert value '{value}' for {key}. Keeping the original value.")

        return hdul






























class AstroqueryClient:
    """
    Old Astroquery Client
    TODO: deprecate everything in this class and use AstroqueryClient ABC instead
    """

    def __init__(self, 
                 query_source="ned",
                 image_source="skyview",
                 root_dir=None,
                 ):

        

        

        self.res = Table()

        self.query_source = query_source.lower()
        self.image_source = image_source.lower()
        self.images = []

        timeout = 60 * 2 # minutes

        Ned.TIMEOUT = timeout
        Simbad.timeout = timeout

        if root_dir is None:
            self.root_dir = os.getcwd()
        else:
            self.root_dir = root_dir

    
    

    def query_galaxies(self, num):

        query = \
        """
        -- Query all SDSS objects
        SELECT TOP 1000 * 
        FROM basic b JOIN otypedef o ON b.otype = o.otype 

        WHERE main_id LIKE 'SDSS%' AND
        (o.otype_longname = 'Galaxy')

        ;
        """
        entire_res = Simbad.query_tap(query)

        self.res = entire_res[np.random.choice(len(entire_res), num, replace=False)]

        return "simbad", self.res

    def query_simbad_sdss(self, search_term=None, split=True):

        query = \
        """
        SELECT TOP 1 * 
        FROM basic b JOIN otypedef o ON b.otype = o.otype 

        WHERE main_id LIKE '%SDSS%' 
        {search}

        ;
        """

        search = ""
        if split:
            for term in search_term.split(" "):
                search += f"AND main_id LIKE '%{term}%'\n"
        else:
            search += f"AND main_id LIKE '%{search_term}%'\n"

        query_formatted = query.format(search=search)
        print(f"Searching {query_formatted}")
        entire_res = Simbad.query_tap(query_formatted)

        self.res = entire_res

        return "simbad", self.res
    
    def query_sdss_by_redshift(self, num=100, z_low=0.1, z_high=0.2):

        # ADQL: galaxies with redshift {z_low} < z < {z_high}
        query = \
        """
        SELECT TOP {num}
            p.objid, s.specobjid, s.z, s.zErr, p.ra, p.dec, s.class
        FROM SpecObj AS s
        JOIN PhotoObj AS p ON s.bestObjID = p.objID
        WHERE s.z BETWEEN {z_low} AND {z_high}
        AND s.class = 'GALAXY'
        ORDER BY s.z
        """

        query_formatted = query.format(num=num, z_low=z_low, z_high=z_high)

        res = SDSS.query_sql(query_formatted)

        coords = SkyCoord(ra=res['ra'], dec=res['dec'], unit=(units.deg, units.deg))
        Simbad.TIMEOUT = 120
        Simbad.add_votable_fields('otypes', 'redshift')

        resolved = []
        for c in coords:
            simbad_res = Simbad.query_region(c, radius='2m')  # match radius
            if simbad_res is not None and len(simbad_res) > 0:
                print([key for key in simbad_res.keys()])
                print(f"len={len(simbad_res)}")
                resolved.append(simbad_res['main_id'][0])
            else:
                resolved.append(None)

        res['main_id'] = resolved

        self.res = res

        return "sdss", self.res

    def query_region(
            self,
            astro_obj=None,
            radius=None
        ):
        '''
        Query Region by coordinate/radius

        '''
        ra_hms, dec_dms = AstroqueryClient.get_ra_dec_in_hms(astro_obj['ra_dec'])
        astro_obj["ra"] = ra_hms
        astro_obj["dec"] = dec_dms

        coordinates = SkyCoord(
            ra=astro_obj["ra"],
            dec=astro_obj["dec"],
            unit=(units.hourangle, units.deg),
            frame='icrs'
        )
        if radius is None:
            radius = 5 * units.arcsec
        if coordinates is None:
            raise ValueError("Coordinate is None")

        print(f"Querying coord {coordinates}, radius {radius}")

        if self.query_source in ["simbad","sdss","fermi"]:

            # self.res = Simbad.query_region(coordinates=coordinates, radius=radius)

            # -- ra >= {coordinates.ra.deg - 1} AND
            # -- ra < {coordinates.ra.deg + 1} AND
            # -- dec >= {coordinates.dec.deg - 1} AND
            # -- dec <= {coordinates.dec.deg + 1} AND

            query = f"""
            -- Query all SDSS objects
            SELECT TOP 10 
            *
            FROM basic b JOIN otypedef o ON b.otype = o.otype 
            WHERE 

            1=CONTAINS(
                POINT('ICRS', ra, dec),
                CIRCLE('ICRS', {coordinates.ra.deg}, {coordinates.dec.deg}, {radius.to(units.deg).value}) 
            ) AND

            (o.otype_longname = 'Galaxy') AND
            (b.morph_type IS NOT NULL) AND
            1=1
            ;
            """
            print(f"Querying SDSS with: {query}")

            custom_criteria = "otype = 'Galaxy'"

            custom_simbad = Simbad()
            custom_simbad.add_votable_fields('morphtype')  # morphology
            custom_simbad.add_votable_fields('otype')      # object type

            # query region
            result = custom_simbad.query_region(coordinates=coordinates, radius=radius, criteria=custom_criteria)

            self.res = result
            self.res.write(f"{self.root_dir}/query_result.csv", format='csv', overwrite=True)


            # self.res.write(f"{self.root_dir}/query_result.csv", format='csv', overwrite=True)

        else:
            raise ValueError(f"Unknown query_source: {self.query_source}")

        print(f"res: {self.res}\n\n")
        # print(f"res.meta: {self.res.meta}\n\n")
        # print(f"res.dtype: {self.res.dtype}\n\n")

        return self.query_source, self.res
    
    def get_images(self):

        # get object name from results
        c = 0
        max_num = 10

        # each row corresponds to one astronomical object.
        # each object might have many images assoicated with many surveys
        for row in self.res:
            
            if c > max_num:
                break
            
            delay = 0.6
            print(f"sleeping {delay} s...")
            sleep(delay)

            fetched_images = []

            all_surveys = {
                "sdss": [
                    'SDSSu',
                    'SDSSg',
                    'SDSSr',
                    'SDSSi',
                    'SDSSz',
                    ],
                "fermi": [
                    'Fermi 5',
                ],
                "ned": [],
            }
            survey = all_surveys[self.image_source]

            obj_name = None
            if self.query_source == "ned":
                obj_name = row['Object Name']
            else:
                obj_name = row['main_id']
                if 'main_id' in row:
                    obj_name = row['main_id']
                if obj_name is None:
                    print(f"obj_name {obj_name} is None for {row['main_id']}")
                    continue
                else:
                    print(f"obj_name {obj_name} is not None")

            names = [(obj_name + s, s) for s in survey]
            for name, s in names:

                fits_filename = f"{self.root_dir}/images/{self.image_source}/fits/{name}.fits"

                if os.path.exists(fits_filename):
                    print(f"Found existing image for {fits_filename}")
                    # get survey from name
                    hdul = [fits.open(fits_filename)]
                    image_obj = {
                        "name": name,
                        "fetched_image": hdul
                    }
                    self.images.append(image_obj)
                    print(f"{s} already exists, removing...")
                    survey.remove(s)
                else:
                    print(f"No image for '{fits_filename}'")
            
            
            debug_header = None
            
            try:
                
                '''
                Source Switch
                '''
                if self.image_source in ["ned"]: # NED
                    
                    print(f"NED images... for {obj_name}...\n")
                    fetched_images = Ned.get_images(obj_name)
                    # redshifts = Ned.get_table(obj_name, table='redshifts')
                
                elif self.image_source in ["simbad", "sdss"]: # SDSS/SIMBAD

                    print(f"Simbad/SDSS images... for {obj_name} and surveys {survey}...\n")
                    fetched_images = SkyView.get_images(position=obj_name, survey=survey)

                elif self.image_source in ["fermi"]: # FERMI

                    print(f"Fermi images.. for {obj_name} and surveys {survey}...\n")
                    fetched_images = SkyView.get_images(position=obj_name, survey=survey)

                else:
                    raise ValueError("Image source not found.")

                print(f"Fetched {len(fetched_images)} image(s) for {obj_name}")

                d = 0
                for hdul in fetched_images:
                    if (len(hdul) == 1):

                        hdul_fixed = hdul
                        hdul_fixed = self.validate_hdul(hdul)

                        # if ('OBS NAME' in hdul_fixed[0].header):
                        #     hdul_fixed[0].header['OBS-NAME'] = hdul_fixed[0].header['OBS NAME']
                        #     del hdul_fixed[0].header['OBS NAME']
                        # if 'EXTEND' in hdul_fixed[0].header:
                        #     extend_value = hdul_fixed[0].header['EXTEND']
                        #     if extend_value is None or extend_value not in [True, False, 'T', 'F']:
                        #         # print(f"Invalid EXTEND value: {extend_value}. Setting EXTEND to True.")
                        #         hdul_fixed[0].header['EXTEND'] = True  # Set to True if extensions exist
                        # else:
                        #     # print("Adding EXTEND keyword with value True.")
                        #     hdul_fixed[0].header['EXTEND'] = True  # Add the keyword if it doesn't exist

                        if 'SURVEY' in hdul_fixed[0].header:
                            image_name = obj_name + "__" + hdul_fixed[0].header["SURVEY"]
                        else:
                            image_name = obj_name + f"__image{d}"
                        d += 1

                        # debug
                        hdul[0].header.totextfile(f"{self.root_dir}/images/{image_name}_header.txt", overwrite=True)
                        debug_header = hdul[0].header

                        image_obj = {
                            "name": image_name,
                            "class": 'unknown',
                            "fetched_image": hdul_fixed
                        }

                        print(f"image_obj: {image_obj}")

                        if hdul_fixed[0].data is None:
                            print(f"Warning: No data in FITS for {image_name}")
                            continue
                        hdul_fixed[0].writeto(f"{self.root_dir}/images/{self.image_source}/fits/{image_name}.fits", 
                                              overwrite=True,
                                              output_verify='fix')

                        self.images.append(image_obj)


            except Exception as e:
                print(f"Error getting images for {obj_name}: {e}\n")
                # debug_header.totextfile(f"{self.root_dir}/images/{image_name}_header.txt", overwrite=True)
                # raise e

        c += 1
        
        return self.image_source, self.images

    
    
    
    def save_images_as_PIL(self):

        os.makedirs(f"{self.root_dir}/images/{self.image_source}/png_pil", exist_ok=True)

        for i in range(len(self.images)):

            image_obj = self.images[i]
            name = image_obj["name"]
            fetched_images = image_obj["fetched_image"]
            image_class = 'unlabeled'
            if image_obj['class'] is not None and image_obj['class'] != '':
                image_class = image_obj['class']

            for image in fetched_images:
                data = image.data
               
                data = np.squeeze(data)

                masked_image = simple_segmentation(data, nsigma=0.5, min_area=10)
                res, _ = morfomytry(image=data)
                data = res['mask'] * data

                # Normalize data to 0-255
                data_min = np.min(data)
                data_max = np.max(data)
                norm_data = (data - data_min) / (data_max - data_min) * 255.0
                # norm_data = data
                # norm_data = norm_data.astype(np.uint8)

                pil_image = Image.fromarray(norm_data)
                pil_image = pil_image.convert("L")  # Convert to grayscale

                arr = np.array(pil_image)
                # Find nonzero region
                nonzero = np.argwhere(arr > 0)
                if nonzero.size > 0:
                    (ymin, xmin), (ymax, xmax) = nonzero.min(0), nonzero.max(0)
                    cropped = pil_image.crop((xmin, ymin, xmax + 1, ymax + 1))
                else:
                    cropped = pil_image  # if all pixels are zero, keep original

                # choose random class from ['A', 'B', 'C']
                random_class = np.random.choice(['A', 'B', 'C', 'test'])

                # make directory if not exists
                os.makedirs(f"{self.root_dir}/images/{self.image_source}/png_pil/{image_class}", exist_ok=True)
                cropped.save(f"{self.root_dir}/images/{self.image_source}/png_pil/{image_class}/{name}.png")

                # if (random_class == 'test'):
                #     cropped.save(f"{self.root_dir}/images/{self.image_source}/png_pil_test/{name}.png")
                # else:
                #     cropped.save(f"{self.root_dir}/images/{self.image_source}/png_pil/{random_class}/{name}.png")

    def plot_images(self):

        # plot with matplotlib
        # print(self.images)

        os.makedirs(f"{self.root_dir}/images/{self.image_source}/png", exist_ok=True)

        for i in range(len(self.images)):

            image_obj = self.images[i]
            name = image_obj["name"]
            fetched_images = image_obj["fetched_image"]

            for image in fetched_images:
                data = image.data
                data = np.squeeze(data)
                norm = ImageNormalize(data, interval=PercentileInterval(99.5), stretch=AsinhStretch())

                plt.figure(figsize=(8, 8))
                plt.imshow(data, norm=norm, origin='lower', cmap='viridis')
                plt.colorbar()
                plt.title(f"{name}")
                plt.savefig(f"{self.root_dir}/images/{self.image_source}/png/{name}.png")
                plt.close()
    
    
