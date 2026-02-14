
import sys
from astropy.table import Table
import importlib
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

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

    def cross_match_labels_simbad(self, df):
        """
        Cross-match 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns 'coord_ra' and 'coord_dec' for cross-matching. 

        Returns
        -------
        matched_df : pandas.DataFrame
            DataFrame with labeled data. 
        """
        log.info("Cross-matching LSST data.")

        # add label from Simbad crossmatch if available
        for i, row in df.iterrows():

            query = \
            """
            SELECT TOP 1 * 
            FROM basic b JOIN otypedef o ON b.otype = o.otype 
            -- WHERE main_id LIKE 'SDSS%' AND
            WHERE 
            ra >= {ra_min}
            AND ra < {ra_max} 
            AND dec >= {dec_min} 
            AND dec <= {dec_max} 
            AND (o.otype_longname = 'Galaxy' OR o.otype_longname = 'Star')
            -- b.rvz_redshift < 0.05 AND
            -- (b.morph_type IS NOT NULL)

            ;
            """
            query = query.format(
                ra_min=row['coord_ra'] - 0.01, 
                ra_max=row['coord_ra'] + 0.01, 
                dec_min=row['coord_dec'] - 0.01, 
                dec_max=row['coord_dec'] + 0.01,
            )
            res = Simbad.query_tap(query)
            # print(f"Query: {query}, number of results: {len(res) if res is not None else 0}")

            label_index = -1  # default to -1 for unknown
            for match_data in res:
                pass
                #for i in match_data.colnames:
                    #print(f"{i}, {match_data[i]}")
                morph_type = str(match_data['morph_type'])
                print(f"Simbad data found for {match_data['main_id']}. Type: {match_data['otype_longname']}, Morphological type: [{morph_type}]")


                label_index = self.pipeline.dataset.labels._get_label_index(morph_type)
            
            # print("Simbad: ")
            # print(res)
            df.at[i, 'label'] = label_index
        return df

    def cross_match_labels_hst(self, df, labels_fits_file):
        """
        Cross-match 

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns 'coord_ra' and 'coord_dec' for cross-matching. 
        labels_fits_file : str
            Path to the FITS file containing labels for cross-matching.

        Returns
        -------
        matched_df : pandas.DataFrame
            DataFrame with labeled data. 
        """
        log.info("Cross-matching LSST data.")


        hst_dict = AstroosQueryLSST.load_hst_and_make_labels(labels_fits_file)
        # print("hst_dict")
        # print(hst_dict)

        df = AstroosQueryLSST.attach_hst_labels_to_lsst(df, hst_dict)

        return df

    @staticmethod
    def load_hst_and_make_labels(hst_fits_path: str):
        hst = Table.read(hst_fits_path, hdu=1)

        ra  = np.array(hst["ra"], dtype=float)
        dec = np.array(hst["dec"], dtype=float)

        # Core physical columns (may contain sentinel values)
        z_best = np.array(hst["z_best"], dtype=float) if "z_best" in hst.colnames else None
        z_spec = np.array(hst["z_spec"], dtype=float) if "z_spec" in hst.colnames else None
        lmass  = np.array(hst["lmass"], dtype=float)  if "lmass" in hst.colnames else None
        lsfr   = np.array(hst["lsfr"], dtype=float)   if "lsfr" in hst.colnames else None
        lssfr  = np.array(hst["lssfr"], dtype=float)  if "lssfr" in hst.colnames else None
        Av     = np.array(hst["Av"], dtype=float)     if "Av" in hst.colnames else None

        # Basic validity mask
        valid = np.isfinite(ra) & np.isfinite(dec)
        if lssfr is None:
            raise ValueError("hst.fits is missing required column 'lssfr'.")

        valid &= np.isfinite(lssfr)

        # Optional: avoid obvious garbage values if present
        # (catalogs sometimes use -99, 0, etc. for missing)
        valid &= (lssfr > -50) & (lssfr < 50)

        # Define labels: 1 = star-forming, 0 = quiescent, -1 = ambiguous
        label = np.full(len(hst), -1, dtype=np.int64)
        label[(valid) & (lssfr < -11.0)] = 0
        label[(valid) & (lssfr > -10.0)] = 1

        # High-confidence subset mask (drop ambiguous)
        confident = (label >= -1)

        return {
            "table": hst,
            "ra": ra,
            "dec": dec,
            "label": label,
            "confident": confident,
            "z_best": z_best,
            "z_spec": z_spec,
            "lmass": lmass,
            "lsfr": lsfr,
            "lssfr": lssfr,
            "Av": Av,
        }

    @staticmethod
    def attach_hst_labels_to_lsst(df_lsst: pd.DataFrame, hst_dict, radius_arcsec=0.8):
        lsst_c = SkyCoord(df_lsst["coord_ra"].to_numpy()*u.deg,
                          df_lsst["coord_dec"].to_numpy()*u.deg)
        hst_c  = SkyCoord(hst_dict["ra"]*u.deg, hst_dict["dec"]*u.deg)


        # print("lsst_c: ", lsst_c)
        # print("hst_c: ", hst_c)

        idx, sep2d, _ = lsst_c.match_to_catalog_sky(hst_c)
        sep_arcsec = sep2d.to(u.arcsec).value
        # print(f"sep_arsec: {sep_arcsec}, radius: {radius_arcsec}")

        # Require a close match AND a confident label
        m = (sep_arcsec <= radius_arcsec) & (hst_dict["confident"][idx])

        out = df_lsst.loc[m].copy()
        hix = idx[m]

        out["hst_sep_arcsec"] = sep_arcsec[m]
        out["label"] = hst_dict["label"][hix].astype(np.int64)   # 0=Q, 1=SF
        out["hst_phot_id"] = np.array(hst_dict["table"]["phot_id"][hix])

        # Optional extra “definitive” aux targets/features
        if hst_dict["z_best"] is not None:
            out["z_best"] = hst_dict["z_best"][hix]
        if hst_dict["z_spec"] is not None:
            out["z_spec"] = hst_dict["z_spec"][hix]
        if hst_dict["lmass"] is not None:
            out["lmass"] = hst_dict["lmass"][hix]
        if hst_dict["lsfr"] is not None:
            out["lsfr"] = hst_dict["lsfr"][hix]
        if hst_dict["lssfr"] is not None:
            out["lssfr"] = hst_dict["lssfr"][hix]
        if hst_dict["Av"] is not None:
            out["Av"] = hst_dict["Av"][hix]

        return out
