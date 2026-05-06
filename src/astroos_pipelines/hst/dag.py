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
    def __init__(
            self,
            dag_dir,
            node_type="catalog_hst",
            node_id=None,
            parents=[],
            parameters=None,
            inputs=[],
            outputs=[],
            origin=True):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label="3D-HST Load Catalog",
            description="Query deep-field cross-survey dataset<br />Provides high-quality SFR labels",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            origin=origin,
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

        columns = {  
                           'ra': "Right Ascension",
                           'dec': "Declination",
                           'jh_mag': "J-H color",
                           'z_spec': "Spectroscopic redshift",
                           'z_peak_phot': "Photometric redshift (peak)",
                           'z_peak_grism': "Grism redshift (peak)",
                           'z_best': "Best redshift",
                           'sfr': "Star Formation Rate",
                           'lssfr': "Log Specific Star Formation Rate",
                           'sfr_IR': "Star Formation Rate from IR",
                           'sfr_UV': "Star Formation Rate from UV",
                           'lmass': "Log Stellar Mass",
                           'Av': "Visual Extinction",
                           'beta': "UV slope",
                           'L_IR': "Infrared Luminosity",
                           'chi2': "Chi-squared of SED fit",
            }
        # pass parameters to next node somehow

        self.output_fits_table(table, columns=columns) # pass the table to the next node, only with the columns we care about for EDA

        

class HSTNodeEDA(Node):
    def __init__(
            self,
            dag_dir,
            node_type="catalog_hst_eda",
            node_id=None,
            parents=[],
            parameters=None,
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label="3D-HST EDA",
            description="Exploratory Data Analysis",
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
        columns = artifact.columns

        dataset_eda(table=table, 
                    columns=columns, 
                    save_dir=self.node_dir, 
                    title="HST")
        
        self.output_fits_table(table, columns=columns) # pass through the table to the next node


class HSTNodeClean(Node):
    def __init__(
            self,
            dag_dir,
            node_type="catalog_hst_clean",
            node_id=None,
            parents=[],
            parameters=None,
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label="3D-HST Clean",
            description="Remove invalid values, clean dataset",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "HSTNodeClean"
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
        print(f"running Clean {self.inputs}")
        artifact = self.inputs[0]
        table = Table.read(artifact.file_path, hdu=1)
        columns = artifact.columns

        df = table.to_pandas()

        bad_map = {
            "Av": [-1],
            "L_IR": [-99],
            "beta": [-99],
            "chi2": [-1],
            "sfr": [-99],
            "sfr_IR": [-99],
            "sfr_UV": [-99],
            "z_best": [-99],
            "z_peak_grism": [-1],
            "z_peak_phot": [-99],
            "z_spec": [-99.9],
        }
        for col, bad_vals in bad_map.items():
            if col in df.columns:
                df[col] = df[col].replace(bad_vals, np.nan) # replace bad values with nan

        numeric_cols = [
            "Av", "L_IR", "beta", "chi2", "dec", "jh_mag", "lmass",
            "ra", "sfr", "sfr_IR", "sfr_UV", "lssfr",
            "z_best", "z_peak_grism", "z_peak_phot", "z_spec"
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") # convert to numeric, set non-convertible values to nan

        if "Av" in df.columns:
            df.loc[df["Av"] < 0, "Av"] = np.nan # Av cannot be negative, so set to nan

        for col in ["z_best", "z_peak_grism", "z_peak_phot", "z_spec"]:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan # redshift cannot be negative, so set to nan

        for col in ["sfr", "sfr_IR", "sfr_UV", "L_IR"]:
            if col in df.columns:
                df.loc[df[col] <= 0, col] = np.nan # SFR and L_IR cannot be negative or zero, so set to nan

        if "ra" in df.columns:
            df.loc[~df["ra"].between(0, 360), "ra"] = np.nan

        if "dec" in df.columns:
            df.loc[~df["dec"].between(-90, 90), "dec"] = np.nan

        all_nan_cols = df.columns[df.isna().all()].tolist()
        df = df.drop(columns=all_nan_cols) # drop columns that are all nan after cleaning

        for col in all_nan_cols:
            if col in columns:
                columns.pop(col, None) # remove all-nan columns from columns dict for next node


        for col in ["L_IR", "sfr", "sfr_IR", "sfr_UV"]:
            if col in df.columns:
                log_col = f"log10_{col}"
                df[log_col] = np.where(df[col] > 0, np.log10(df[col]), np.nan) # add log10 versions of SFR and L_IR, set to nan if original value is not positive
                columns[log_col] = f"log10 {col}" # add log versions of SFR and L_IR to the columns dict for the next node
                # remove from dict
                columns.pop(col, None) # remove original column from columns dict, since we'll use the log version for EDA

        for col in ["chi2", "L_IR", "sfr", "sfr_IR", "sfr_UV"]:
            if col in df.columns:
                q999 = df[col].quantile(0.999)
                df[f"{col}_is_extreme"] = df[col] > q999

        clean_table = Table.from_pandas(df)

        self.output_fits_table(clean_table, columns=columns) # pass the cleaned table to the next node, only with the columns we care about for EDA



class HSTNodeSelect(Node):
    def __init__(
            self,
            dag_dir,
            node_type="catalog_hst_select",
            node_id=None,
            parents=[],
            parameters=None,
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            label="3D-HST Data Selection",
            description="Remove outliers",
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "HSTNodeSelect"
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
        print(f"running Clean {self.inputs}")
        artifact = self.inputs[0]
        table = Table.read(artifact.file_path, hdu=1)
        columns = artifact.columns

        df = table.to_pandas()

        def clip_outliers(series, lower=0.001, upper=0.999):
            lo = series.quantile(lower)
            hi = series.quantile(upper)
            return series.clip(lo, hi)

        df["beta"] = clip_outliers(df["beta"]) # beta has extreme outliers, so clip to 0.1 and 99.9 percentiles
        df = df[df["chi2"] < df["chi2"].quantile(0.99)] # chi2 has extreme outliers, so remove top 1%
        # remove outliers of lssfr
        df = df[df["lssfr"] > df["lssfr"].quantile(0.05)] # remove bottom 5% of lssfr for visualization

        science_table = Table.from_pandas(df)

        self.output_fits_table(science_table, columns=columns) # pass the selected table to the next node, only with the columns we care about for EDA

