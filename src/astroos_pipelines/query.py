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

# astropy
from astropy.coordinates import SkyCoord
from astropy import units
from astropy.table import Table
from astropy.visualization import (ImageNormalize, PercentileInterval, AsinhStretch)
from astropy.io import fits

from abc import ABC, abstractmethod
import importlib

from astroos_pipelines.morphometry import simple_segmentation, measure_morfometry, morfomytry
from astroos_pipelines.tap_clients import PyvoTAPClient

importlib.reload(sys.modules['astroos_pipelines.morphometry'])
importlib.reload(sys.modules['astroos_pipelines.tap_clients'])


# ============================================================
# AstroosQuery Abstract Base Class
# ============================================================
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
# AstroosQuery SDSS
# ============================================================
class AstroosQuerySDSS(AstroosQuery):
    """
    Astroquery SDSS Client
    See https://skyserver.sdss.org/dr16/en/help/docs/api.aspx#advanced for details.

    """

    def __init__(self, root_dir):
        super().__init__( 
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
