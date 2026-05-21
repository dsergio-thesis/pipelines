import matplotlib.gridspec as gridspec

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
# from astroos_pipelines.pipelines import StagePipeline 
from astroos_pipelines.dag import *
from astroos_pipelines.utils.rsp import get_cutout_bands
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.utils.plots.dataset_eda import dataset_eda

# importlib.reload(sys.modules['astroos_pipelines.dag'])
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])
importlib.reload(sys.modules['astroos_pipelines.utils.rsp'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.dataset_eda'])
importlib.reload(sys.modules['astroos_pipelines.query'])
importlib.reload(sys.modules['astroos_pipelines.datasets'])

import matplotlib.pyplot as plt
import seaborn as sns


from astropy.io import fits
# do wcs next

rsp_mode = False
try:
    from lsst.rsp import get_tap_service
    from lsst.rsp.utils import get_pyvo_auth
    from lsst.rsp.service import get_siav2_service
    from lsst.rsp.utils import get_pyvo_auth
    import lsst.geom as geom
    from lsst.afw.fits import MemFileManager

    # other LSST dependencies
    from pyvo.dal.adhoc import DatalinkResults
    from astropy.time import Time
    from pyvo.dal.adhoc import DatalinkResults, SodaQuery
    rsp_mode = True

except ImportError as e:
    print(f"LSST RSP dependencies not found. RSP mode will be disabled. Please install the required packages: {e}")
    pass

# from astroos_pipelines.logger.logger import setup_logging
# importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
# import logging
# setup_logging()
# log = logging.getLogger(__name__)


# class LSSTNodeCatalog(Node):
    # def __init__(self,
                # dag_dir=None,
                # node_type="catalog_lsst",
                # node_id=None,
                # parents=[],
                # parameters=None,
                # origin=True,
                # ):
        # super().__init__(
            # node_type=node_type,
            # dag_dir=dag_dir,
            # label="Query LSST DP-1 TAP",
            # description="Query the Table Access Protocol (TAP)<br />for photometric features",
            # node_id=node_id,
            # parents=parents,
            # parameters=parameters,
            # origin=origin,
            # )

    # def to_dict(self):
        # d = super().to_dict()
        # d["type"] = "LSSTNodeCatalog"
        # return d

    # @classmethod
    # def _from_dict(cls, d):
        # return cls(
            # node_id=d["node_id"],
            # parents=d.get("parents", []),
            # parameters=d.get("parameters", {}),
            # inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            # outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        # )

    # def run(self):

        # query_coords = self.parameters.get('query_coords')
        # query_radius = self.parameters.get('query_radius')
        # max_records = self.parameters.get('max_records')

        # client = AstroosQueryLSST(root_dir=f"_pipelines/{self.node_id}", 
                                  # credentials_file=None,
                                  # max_records=max_records)

        # if query_radius > 0:
            # dec_min = max(query_coords.dec.deg - query_radius.to(u.deg).value, -90)
            # dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

            # delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
            # ra_min = (query_coords.ra.deg - delta_ra) % 360
            # ra_max = (query_coords.ra.deg + delta_ra) % 360


            # # Query for objects within the RA/Dec box defined by the center and radius.
            # query = \
            # """
            # SELECT TOP {max_records}
            # objectId,
            # tract,
            # patch,
            # coord_ra,
            # coord_dec,

            # detect_fromBlend, detect_isIsolated,

            # -- u
            # u_psfFlux,            u_psfFluxErr,            u_psfFlux_flag,
            # u_free_cModelFlux,    u_free_cModelFluxErr,    u_free_cModelFlux_flag,

            # -- g
            # g_psfFlux,            g_psfFluxErr,            g_psfFlux_flag,
            # g_free_cModelFlux,    g_free_cModelFluxErr,    g_free_cModelFlux_flag,

            # -- r
            # r_psfFlux,            r_psfFluxErr,            r_psfFlux_flag,
            # r_free_cModelFlux,    r_free_cModelFluxErr,    r_free_cModelFlux_flag,

            # -- i
            # i_psfFlux,            i_psfFluxErr,            i_psfFlux_flag,
            # i_free_cModelFlux,    i_free_cModelFluxErr,    i_free_cModelFlux_flag,

            # -- z
            # z_psfFlux,            z_psfFluxErr,            z_psfFlux_flag,
            # z_free_cModelFlux,    z_free_cModelFluxErr,    z_free_cModelFlux_flag,

            # -- y
            # y_psfFlux,            y_psfFluxErr,            y_psfFlux_flag,
            # y_free_cModelFlux,    y_free_cModelFluxErr,    y_free_cModelFlux_flag,

            # refExtendedness

            # FROM dp1.Object
            
            # WHERE coord_ra BETWEEN {ra_min} AND {ra_max}
                # AND coord_dec BETWEEN {dec_min} AND {dec_max}

            # """
            # query = query.format(
                    # max_records=max_records,
                    # ra_min=ra_min,
                    # ra_max=ra_max,
                    # dec_min=dec_min,
                    # dec_max=dec_max
                    # )
        # else:

            # # Query for all objects (up to max_records limit)
            # query = \
            # """
            # SELECT TOP {max_records}
            # objectId,
            # tract,
            # patch,
            # coord_ra,
            # coord_dec,

            # detect_fromBlend, detect_isIsolated,

            # -- u
            # u_psfFlux,            u_psfFluxErr,            u_psfFlux_flag,
            # u_free_cModelFlux,    u_free_cModelFluxErr,    u_free_cModelFlux_flag,

            # -- g
            # g_psfFlux,            g_psfFluxErr,            g_psfFlux_flag,
            # g_free_cModelFlux,    g_free_cModelFluxErr,    g_free_cModelFlux_flag,

            # -- r
            # r_psfFlux,            r_psfFluxErr,            r_psfFlux_flag,
            # r_free_cModelFlux,    r_free_cModelFluxErr,    r_free_cModelFlux_flag,

            # -- i
            # i_psfFlux,            i_psfFluxErr,            i_psfFlux_flag,
            # i_free_cModelFlux,    i_free_cModelFluxErr,    i_free_cModelFlux_flag,

            # -- z
            # z_psfFlux,            z_psfFluxErr,            z_psfFlux_flag,
            # z_free_cModelFlux,    z_free_cModelFluxErr,    z_free_cModelFlux_flag,

            # -- y
            # y_psfFlux,            y_psfFluxErr,            y_psfFlux_flag,
            # y_free_cModelFlux,    y_free_cModelFluxErr,    y_free_cModelFlux_flag,

            # refExtendedness

            # FROM dp1.Object
            # """

            # query = query.format(
                    # max_records=max_records,
                    # )

        # # print(f"Query: {query}")
        
        # # sync
        # # table = client.query(query)

        # # async
        # print("Running TAP ADQL Query on LSST...")
        # table = client.query_async(query)


        # columns = {}
        # for col in table.colnames:
            # columns[col] = col
        # # columns.pop("objectId", None)


        # if len(self.inputs) > 0:
            # artifact = self.inputs[0]
            # artifact.load_from_table(table, columns)

            # self.outputs = [artifact]


        # # self.output_fits_table(table, columns=columns)
        # # print(f"number of results: {len(table)}")



class NodeTAPQuery(Node):
    """
    A node that connects to a TAP service.

    """

    def __init__(self,
                 dag_dir=None,
                 node_type="NodeTAPQuery",
                 node_id=None,
                 parents=[],
                 parameters={"script": None},
                 origin=True,
                 label="TAP Query",
                 inputs=[],
                 outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            label=label,
            inputs=inputs,
            outputs=outputs,
            origin=origin,
            description="A node that connects to a TAP service.",
        )
    
    def node_configure(self):
        if self.parameters['script'] is None:
            # write template script to node directory
            template_script = """# Example script for NodeTAPQuery

query["description"] = "Get 10 random objects"
query["adql"] = "SELECT TOP 10 objectId FROM dp1.Object"

"""         
            script_path = os.path.join(self.node_dir, f"script.py")

            os.makedirs(self.node_dir, exist_ok=True)
            with open(script_path, "w") as f:
                f.write(template_script)
            self.parameters = {
                    "script": script_path
                }
    
    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeTAPQuery"
        return d
    
    @classmethod
    def _from_dict(cls, d):
        return cls(
            node_id=d["node_id"],
            dag_dir=d["dag_dir"],
            parents=d.get("parents", []),
            parameters=d.get("parameters", {}),
            inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        )

    def run(self):

        script = self.parameters.get("script", "")
        max_records = self.parameters.get("max_records", 3)

        query = {"adql": "", "description": ""}

        with open(script, "r") as f:
            code = f.read()
            exec(code, {"query": query, 
                        "parameters": self.parameters, 
                        })

        client = AstroosQueryLSST(root_dir=f"_pipelines/{self.node_id}", 
                      credentials_file=None,
                      max_records=max_records)

        print("Running TAP ADQL Query on LSST...")
        table = client.query_async(query["adql"])
        
        print(f"Number of results: {len(table)}")

        columns = {}
        for col in table.colnames:
            columns[col] = col
        # columns.pop("objectId", None)


        artifact = ArtifactItem(
                file_path=os.path.join(self.node_dir, "tap.fits"),
                dag=self.artifact_dag,
                node_id=self.node_id,
                )
        artifact.load_from_table(table, columns)
        artifact.materialize(self.node_id)

        self.outputs = [artifact]



# class LSSTNodeEDA(Node):
    # def __init__(
            # self,
            # node_type="lsst_eda",
            # node_id=None,
            # parents=[],
            # parameters=None,
            # inputs=[],
            # outputs=[]):
        # super().__init__(
            # node_type=node_type,
            # label="LSST DP-1 EDA",
            # description="LSST DP-1 Exploratory Data Analysis",
            # node_id=node_id,
            # parents=parents,
            # parameters=parameters,
            # inputs=inputs,
            # outputs=outputs,
            # )

    # def to_dict(self):
        # d = super().to_dict()
        # d["type"] = "LSSTNodeEDA"
        # return d

    # @classmethod
    # def _from_dict(cls, d):
        # return cls(
            # node_id=d["node_id"],
            # parents=d.get("parents", []),
            # parameters=d.get("parameters", {}),
            # inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            # outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        # )

    # def run(self):

        # artifact = self.inputs[0]
        # table = Table.read(artifact.file_path, hdu=1)
        # columns = artifact.columns

        # dataset_eda(table=table,
                    # columns=columns,
                    # save_dir=f"_pipelines/{self.node_id}",
                    # title="LSST DP1")

        # self.output_fits_table(table, columns=columns)


# class LSSTNodeMatchToHST(Node):
    # def __init__(
            # self,
            # dag_dir=None,
            # node_type="catalog_lsst_match_hst",
            # node_id=None,
            # parents=[],
            # parameters=None,
            # inputs=[],
            # outputs=[],
            # ):
        # super().__init__(
            # node_type=node_type,
            # dag_dir=dag_dir,
            # label="LSST DP-1 HST-3D Cross-match",
            # description="Use Deep-field HST-3D labels for supervised learning.",
            # node_id=node_id,
            # parents=parents,
            # parameters=parameters,
            # num_inputs=2,
            # num_outputs=1,
            # inputs=inputs,
            # outputs=outputs,
            # )

    # def to_dict(self):
        # d = super().to_dict()
        # d["type"] = "LSSTNodeMatchToHST"
        # return d

    # @classmethod
    # def _from_dict(cls, d):
        # return cls(
            # node_id=d["node_id"],
            # parents=d.get("parents", []),
            # parameters=d.get("parameters", {}),
            # inputs=[ArtifactItem.from_dict(a) for a in d.get("inputs", [])],
            # outputs=[ArtifactItem.from_dict(a) for a in d.get("outputs", [])],
        # )

    # def run(self):

        # return

        # # expects 2 input artifacts: LSST catalog and HST catalog (both as FITS tables with RA/Dec columns)
        # if len(self.inputs) < 2:
            # print(f"LSSTNodeMatchToHST self.inputs: {self.inputs}")
            # raise RuntimeError("LSSTNodeMatchToHST requires 2 input artifacts named `catalog_hst_select` and `catalog_lsst`.")
        
        # hst_table = Table()
        # lsst_table = Table()
        # for artifact in self.inputs:
            # if (artifact.name == "catalog_hst_select"):
                # hst_table = Table.read(artifact.file_path, hdu=1)
            # elif (artifact.name == "catalog_lsst"):
                # lsst_table = Table.read(artifact.file_path, hdu=1)

        # if (len(hst_table) == 0 or len(lsst_table) == 0):
            # print(f"Either HST or LSST tables are size 0. self.inputs: {self.inputs}")
            # raise RuntimeError(f"Either HST or LSST tables are size 0. LSSTNodeMatchToHST requires 2 input artifacts named `catalog_hst_select` and `catalog_lsst`.")

        # table = Table.from_pandas(
                # AstroosQueryLSST.cross_match_labels_hst(
                    # lsst_table.to_pandas(), 
                    # hst_table.to_pandas(), 
                    # max_sep_arcsec=1.0))

        # # print("pipeline labels match: ")
        # # print(self.output['label'].value_counts())

        # self.output_fits_table(table)

# class LSSTNodePreprocess(Node):
    # def __init__(
            # self,
            # dag_dir,
            # node_type="catalog_lsst_preprocess",
            # node_id=None,
            # parents=[],
            # parameters=None,
            # inputs=[],
            # outputs=[]):
        # super().__init__(
            # node_type=node_type,
            # dag_dir=dag_dir,
            # label="Extract LSST Photometric features",
            # description="Extract colors from photometry.",
            # node_id=node_id,
            # parents=parents,
            # parameters=parameters,
            # inputs=inputs,
            # outputs=outputs,
            # )

    # def to_dict(self):
        # d = super().to_dict()
        # d["type"] = "LSSTNodePreprocess"
        # return d

    # @classmethod
    # def _from_dict(cls, d):
        # return cls(
            # node_id=d["node_id"],
            # parents=d.get("parents", []),
            # parameters=d.get("parameters", {}),
            # inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            # outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        # )

    # def run(self):
        # """
        # 3 features per band: 
            # - flux Transformed (arcsinh)
            # - x err Transformed (arcsinh)
            # - log SNR (clamped to 0 if err=0)
            # - mag (from flux, with safe handling of zero/negative flux)
            # - x bad-flag (1 if any issues with flux/err, else 0)

        # 15 color features:
            # - u-g color (mag_u - mag_g)
            # - u-r color (mag_u - mag_r)
            # - u-i color (mag_u - mag_i)
            # - u-z color (mag_u - mag_z)
            # - u-y color (mag_u - mag_y)

            # - g-r color (mag_g - mag_r)
            # - g-i color (mag_g - mag_i)
            # - g-z color (mag_g - mag_z)
            # - g-y color (mag_g - mag_y)

            # - r-i color (mag_r - mag_i)
            # - r-z color (mag_r - mag_z)
            # - r-y color (mag_r - mag_y)
            
            # - i-z color (mag_i - mag_z)
            # - i-y color (mag_i - mag_y)

            # - z-y color (mag_z - mag_y)

        # 4 Adjacent curvatures:
            # - curv_ug_gr = (mag_u - mag_g) - (mag_g - mag_r) = mag_u - 2*mag_g + mag_r
            # - curv_gr_ri = (mag_g - mag_r) - (mag_r - mag_i) = mag_g - 2*mag_r + mag_i
            # - curv_ri_iz = (mag_r - mag_i) - (mag_i - mag_z) = mag_r - 2*mag_i + mag_z
            # - curv_iz_zy = (mag_i - mag_z) - (mag_z - mag_y) = mag_i - 2*mag_z + mag_y

        # Next: add diff/ratio PSF and cModel for morphology

        # """

        # artifact = self.inputs[0]
        # table = Table.read(artifact.file_path, hdu=1)
        # df = table.to_pandas()

        # df_clean = pd.DataFrame()  # will hold cleaned data with new features
        # df_clean['objectId'] = df['objectId']
        # df_clean['ra'] = df['coord_ra']
        # df_clean['dec'] = df['coord_dec']
        # df_clean['tract'] = df['tract']
        # df_clean['patch'] = df['patch']
        # df_clean['detect_fromBlend'] = df['detect_fromBlend']
        # df_clean['detect_isIsolated'] = df['detect_isIsolated']
        # df_clean['refExtendedness'] = df['refExtendedness']
        # df_clean['label'] = df['label'] if 'label' in df.columns else [np.nan] * len(df)
        # df_clean['color_gr'] = [np.nan] * len(df)
        # df_clean['color_ri'] = [np.nan] * len(df)
        # df_clean['color_iz'] = [np.nan] * len(df)
        # for band in ['u', 'g', 'r', 'i', 'z', 'y']:
            # df_clean[f"{band}_psfFlux_arcsinh"] = [np.nan] * len(df)
            # df_clean[f"{band}_psfFluxErr_arcsinh"] = [np.nan] * len(df)
            # df_clean[f"{band}_psfFlux_SNR_log"] = [np.nan] * len(df)
            # df_clean[f"{band}_psfFlux_mag"] = [np.nan] * len(df)
            # df_clean[f"{band}_psfFlux_bad_flag"] = [np.nan] * len(df)

        # n = len(df)
        # print(f"Feature preprocesing for {n} objects...")

        # def flux_to_mag(flux):
            # return -2.5 * np.log10(flux) + 31.4

        # bands = ['u', 'g', 'r', 'i', 'z', 'y']

        # label_counts = dict()

        # num_bands = len(bands)

        # # precompute safe scales (dataset-level)
        # flux_scale = self.median_r_psfFlux if getattr(self, "median_r_psfFlux", 0) and self.median_r_psfFlux > 0 else 1.0
        # err_scale  = self.median_r_psfFluxErr if getattr(self, "median_r_psfFluxErr", 0) and self.median_r_psfFluxErr > 0 else 1.0

        # for row in tqdm(df.itertuples(), total=n, desc="Extracting Photometric Features"):
            # target_ra = row.coord_ra
            # target_dec = row.coord_dec

            # if hasattr(row, "label"):
                # if (str(row.label) in label_counts):
                    # # print(f"found label {str(row.label)}, adding to existing counts")
                    # label_counts[str(row.label)] += 1
                # else:
                    # # print(f"found label {str(row.label)}, setting count to 1")
                    # label_counts[str(row.label)] = 1

            # photometric_features = np.zeros((num_bands, 3), dtype=np.float32)
            
            # mag_g = None
            # mag_g_flag = True
            # mag_r = None
            # mag_r_flag = True
            # mag_i = None
            # mag_i_flag = True
            # mag_z = None
            # mag_z_flag = True

            # for bi, band in enumerate(bands):
                # flux = getattr(row, f"{band}_psfFlux", None)
                # err  = getattr(row, f"{band}_psfFluxErr", None)
                # flag = getattr(row, f"{band}_psfFlux_flag", False)

                # mag = flux_to_mag(flux)
                
                # # sanitize missing/NaN
                # if flux is None or err is None or (isinstance(flux, float) and np.isnan(flux)) or (isinstance(err, float) and np.isnan(err)):
                    # x1 = 0.0
                    # x2 = 0.0
                    # x3 = 0.0
                    # x4 = 0.0
                    # bad = 1.0  # treat missing as bad
                # else:
                    # # arcsinh scaling 
                    # x1 = np.arcsinh(float(flux) / flux_scale)
                    # x2 = np.arcsinh(float(err) / err_scale)

                    # # SNR feature (clamp to non-negative)
                    # if err > 0:
                        # snr = float(flux) / float(err)
                        # x3 = np.log1p(max(0.0, snr))
                    # else:
                        # x3 = 0.0

                    # x4 = mag
                    # if band == 'g':
                        # mag_g = mag
                        # mag_g_flag = flag
                    # elif band == 'r':
                        # mag_r = mag
                        # mag_r_flag = flag
                    # elif band == 'i':
                        # mag_i = mag
                        # mag_i_flag = flag
                    # elif band == 'z':
                        # mag_z = mag
                        # mag_z_flag = flag
                    # elif band == 'u':
                        # mag_u = mag
                        # mag_u_flag = flag
                    # elif band == 'y':
                        # mag_y = mag
                        # mag_y_flag = flag

                    # bad = 1.0 if bool(flag) else 0.0

                # photometric_features[bi] = (x1, x3, x4)

                # df_clean.at[row.Index, f"{band}_psfFlux_arcsinh"] = x1
                # # df_clean.at[row.Index, f"{band}_psfFluxErr_arcsinh"] = x2
                # df_clean.at[row.Index, f"{band}_psfFlux_SNR_log"] = x3
                # df_clean.at[row.Index, f"{band}_psfFlux_mag"] = x4
                # # df_clean.at[row.Index, f"{band}_psfFlux_bad_flag"] = bad

            # mags = {
                # "u": mag_u,
                # "g": mag_g,
                # "r": mag_r,
                # "i": mag_i,
                # "z": mag_z,
                # "y": mag_y,
            # }

            # flags = {
                # "u": mag_u_flag,
                # "g": mag_g_flag,
                # "r": mag_r_flag,
                # "i": mag_i_flag,
                # "z": mag_z_flag,
                # "y": mag_y_flag,
            # }

            # def color(b1, b2):
                # m1, m2 = mags[b1], mags[b2]
                # f1, f2 = flags[b1], flags[b2]
                # return m1 - m2 if (m1 is not None and m2 is not None and not f1 and not f2) else np.nan
            
            # colors = {}
            # colors['ug'] = color('u', 'g')
            # colors['ur'] = color('u', 'r')
            # colors['ui'] = color('u', 'i')
            # colors['uz'] = color('u', 'z')
            # colors['uy'] = color('u', 'y')
            # colors['gr'] = color('g', 'r')
            # colors['gi'] = color('g', 'i')
            # colors['gz'] = color('g', 'z')
            # colors['gy'] = color('g', 'y')
            # colors['ri'] = color('r', 'i')
            # colors['rz'] = color('r', 'z')
            # colors['ry'] = color('r', 'y')
            # colors['iz'] = color('i', 'z')
            # colors['iy'] = color('i', 'y')
            # colors['zy'] = color('z', 'y')

            # curvatures = {}
            # curvatures['ug_gr'] = (colors['ug'] - colors['gr']) if (not np.isnan(colors['ug']) and not np.isnan(colors['gr'])) else np.nan
            # curvatures['gr_ri'] = (colors['gr'] - colors['ri']) if (not np.isnan(colors['gr']) and not np.isnan(colors['ri'])) else np.nan
            # curvatures['ri_iz'] = (colors['ri'] - colors['iz']) if (not np.isnan(colors['ri']) and not np.isnan(colors['iz'])) else np.nan
            # curvatures['iz_zy'] = (colors['iz'] - colors['zy']) if (not np.isnan(colors['iz']) and not np.isnan(colors['zy'])) else np.nan

            # photometric_features = np.hstack([photometric_features.flatten(), 
                # [color for color in colors.values()] + 
                # [curv for curv in curvatures.values()]
                                              # ])
            # for color_name, color_value in colors.items():
                # df_clean.at[row.Index, f"color_{color_name}"] = color_value
            # for curv_name, curv_value in curvatures.items():
                # df_clean.at[row.Index, f"curvature_{curv_name}"] = curv_value


            # # hdu_phot = fits.ImageHDU(data=photometric_features, name="PHOTO")
            # # hdu_phot.header['label'] = int(row.label) if hasattr(row, "label") else 0
            # # hdu_phot.header['ra'] = float(target_ra)
            # # hdu_phot.header['dec'] = float(target_dec)
            # # hdu_phot.header['objectId'] = int(row.objectId)

            # # dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(self.parameters.get("dataset"))

            # # if (dataset.contains(row.objectId)):
                # # dataset.update(row.objectId, hdu_phot)
            # # else:
                # # dataset.append(hdu_phot)

        # print(f"Label counts: {label_counts}")

        # columns = {}
        # for col in df_clean.columns:
            # columns[col] = col

        # columns.pop("objectId", None)

        # table = Table.from_pandas(df_clean)
        # self.output_fits_table(table, columns=columns)



# class LSSTNodePhotoDataset(Node):
    # def __init__(
            # self,
            # dag_dir,
            # node_type="catalog_lsst_photo_dataset",
            # node_id=None,
            # parents=[],
            # parameters=None,
            # inputs=[],
            # outputs=[]):
        # super().__init__(
            # node_type=node_type,
            # dag_dir=dag_dir,
            # label="Construct FITS Dataset",
            # description="Construct photometric dataset.",
            # node_id=node_id,
            # parents=parents,
            # parameters=parameters,
            # inputs=inputs,
            # outputs=outputs,
            # )

    # def to_dict(self):
        # d = super().to_dict()
        # d["type"] = "LSSTNodePhotoDataset"
        # return d

    # @classmethod
    # def _from_dict(cls, d):
        # return cls(
            # node_id=d["node_id"],
            # parents=d.get("parents", []),
            # parameters=d.get("parameters", {}),
            # inputs=[Artifact.from_dict(a) for a in d.get("inputs", [])],
            # outputs=[Artifact.from_dict(a) for a in d.get("outputs", [])],
        # )

    # def run(self):

        # artifact = self.inputs[0]
        # table = Table.read(artifact.file_path, hdu=1)
        # df = table.to_pandas()
        # columns = artifact.columns

        # dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(self.parameters.get("dataset"))
        # dataset.feature_names = columns

        # for row in tqdm(df.itertuples(), total=len(df), desc="Building Photometric Dataset"):

            # target_ra = row.ra
            # target_dec = row.dec
            # photometric_features = np.zeros((6, 3), dtype=np.float32)
            # for bi, band in enumerate(['u', 'g', 'r', 'i', 'z', 'y']):
                # photometric_features[bi] = [
                    # getattr(row, f"{band}_psfFlux_arcsinh", 0.0),
                    # # getattr(row, f"{band}_psfFluxErr_arcsinh", 0.0),
                    # getattr(row, f"{band}_psfFlux_SNR_log", 0.0),
                    # getattr(row, f"{band}_psfFlux_mag", 0.0),
                    # # getattr(row, f"{band}_psfFlux_bad_flag", 0.0),
                # ]
            # photometric_features = np.hstack([photometric_features.flatten(),
                # getattr(row, "color_ug", np.nan),
                # getattr(row, "color_ur", np.nan),
                # getattr(row, "color_ui", np.nan),
                # getattr(row, "color_uz", np.nan),
                # getattr(row, "color_uy", np.nan),
                # getattr(row, "color_gr", np.nan),
                # getattr(row, "color_gi", np.nan),
                # getattr(row, "color_gz", np.nan),
                # getattr(row, "color_gy", np.nan),
                # getattr(row, "color_ri", np.nan),
                # getattr(row, "color_rz", np.nan),
                # getattr(row, "color_ry", np.nan),
                # getattr(row, "color_iz", np.nan),
                # getattr(row, "color_iy", np.nan),
                # getattr(row, "color_zy", np.nan),
                # getattr(row, "curvature_ug_gr", np.nan),
                # getattr(row, "curvature_gr_ri", np.nan),
                # getattr(row, "curvature_ri_iz", np.nan),
                # getattr(row, "curvature_iz_zy", np.nan),])

            # hdu_phot = fits.ImageHDU(data=photometric_features, name="PHOTO")
            # hdu_phot.header['label'] = int(row.label) if hasattr(row, "label") else 0
            # hdu_phot.header['ra'] = float(target_ra)
            # hdu_phot.header['dec'] = float(target_dec)
            # hdu_phot.header['objectId'] = int(row.objectId)

            # if (dataset.contains(row.objectId)):
                # dataset.update(row.objectId, hdu_phot)
            # else:
                # dataset.append(hdu_phot)

        # table = Table.from_pandas(df)
        # self.output_fits_table(table, columns=self.parameters.get("columns", None))


class NodeLSSTButlerFetch(Node):
    def __init__(
            self,
            dag_dir=None,
            node_type="catalog_lsst_butler_fetch",
            node_id=None,
            parents=[],
            parameters={},
            label="Fetch LSST DP-1 Cutouts (Butler)",
            inputs=[],
            outputs=[]):
        super().__init__(
            node_type=node_type,
            dag_dir=dag_dir,
            label=label,
            description="Use the RSP Butler service <br />to fetch deep coadd cutouts.",
            node_id=node_id,
            parents=parents,
            parameters=parameters,
            inputs=inputs,
            outputs=outputs,
            )

    def to_dict(self):
        d = super().to_dict()
        d["type"] = "NodeLSSTButlerFetch"
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

        artifact = self.inputs[0]
        table = artifact.to_table(self.node_id) 

        print(f"Fetching LSST data via Butler for {len(table)} objects...")

        tasks = build_groups(
                table, 
                self.parameters.get("dataset")
                )

        with ProcessPoolExecutor(max_workers=8) as ex:
            for _ in ex.map(worker_patch, tasks):
                pass



def worker_patch(args):

    BANDS = ["u", "g", "r", "i", "z"]
    BANDS = ["u", "g", "r", "i", "z", "y"]

    # cutout stamp size (pixels)
    STAMP_W = 100
    STAMP_H = 100

    # tract, patch, object_rows, dataset_dir, dataset_labels = args
    tract, patch, object_rows, dataset_dict = args

    # print("object_rows")
    # print(object_rows)

    # dataset = FITS_Image_Morphometry_Photometry_Dataset(
            # dataset_dir=dataset_dir,
            # labels_init_file=dataset_labels,
            # N_bands=len(BANDS), 
            # N_morphometric_features=4,
            # N_photometric_features=4,
            # )
    dataset = FITS_Image_Morphometry_Photometry_Dataset.from_dict(dataset_dict)

    from lsst.daf.butler import Butler
    butler = Butler("dp1", collections="LSSTComCam/DP1")

    # Load each band ONCE per patch
    coadds = {
        b: butler.get("deep_coadd", tract=tract, patch=patch, band=b)
        for b in BANDS
    }

    ext = geom.Extent2I(STAMP_W, STAMP_H)
    band_images = np.zeros((len(BANDS), STAMP_H, STAMP_W), dtype=np.float32)

    for row in object_rows:

        # print("row\n\n")
        # print(row)
        ra_deg = float(row["ra"])
        dec_deg = float(row["dec"])

        # SpherePoint expects (lon, lat) as Angles.
        # Use degrees explicitly.
        sky = geom.SpherePoint(ra_deg * geom.degrees, dec_deg * geom.degrees)

        wcs_header = fits.Header()
        hdr = fits.Header()
        
        min_ra = ra_deg - 0.0138889
        max_ra = ra_deg + 0.0138889
        min_dec = dec_deg - 0.0138889
        max_dec = dec_deg + 0.0138889

        for band, exp in coadds.items():
            wcs = exp.getWcs()
            if wcs is None:
                # Shouldn't happen for coadds, but guard anyway
                continue

            # Convert sky coordinate to pixel coordinate in this exposure
            pix = wcs.skyToPixel(sky)  # returns lsst.geom.Point2D

            # Optional: skip objects whose pixel center is off-image
            bbox = exp.getBBox()
            if not bbox.contains(geom.Point2I(int(round(pix.getX())), int(round(pix.getY())))):
                continue

            cutout = exp.getCutout(pix, ext)
            # get minimal WCS info for the cutout
            wcs_cutout = cutout.getWcs()


            bbox = cutout.getBBox()
            # hdr = fits_header_from_lsst_cutout_wcs(cutout.getWcs(), bbox)
            hdr = make_cutout_header3(cutout, ra_deg, dec_deg)
            # print(f"hdr: {hdr}")


            # print("cutout bbox:", cutout.getBBox())  # should show a small region
            # print("cutout dims:", cutout.getDimensions())  # width/height
            # print("cutout WCS:", wcs_cutout)  # should be a valid WCS object

            if (band == "r"):
                wcs_header = wcs_cutout.getFitsMetadata()

                # min_ra, max_ra = wcs_cutout.getSkyBBox().getMin().getX(), wcs_cutout.getSkyBBox().getMax().getX()
                # min_dec, max_dec = wcs_cutout.getSkyBBox().getMin().getY(), wcs_cutout.getSkyBBox().getMax().getY()
                # min_ra, max_ra, min_dec, max_dec = wcs_bounds_radec(wcs_cutout, STAMP_W, STAMP_H)
            
            band_images[BANDS.index(band)] = cutout.getImage().getArray()

        target_ra = ra_deg 
        target_dec = dec_deg

        
        

        # print(f"band_images shape: {band_images.shape}")

        hdu_img = fits.ImageHDU(data=band_images, name="CUTOUTS")
        hdu_img.header['label'] = int(row['label'])
        hdu_img.header['ra'] = float(target_ra)
        hdu_img.header['dec'] = float(target_dec)
        hdu_img.header['objectId'] = int(row['objectId'])
        # hdu_img.header['redshift'] = -999
        # hdu_img.header['min_ra'] = min_ra
        # hdu_img.header['max_ra'] = max_ra
        # hdu_img.header['min_dec'] = min_dec
        # hdu_img.header['max_dec'] = max_dec

        for k, v in hdr.items():
            hdu_img.header[k] = v
            # print(f"wcs header: {k}: {v}")

        if (dataset.contains(row['objectId'])):
            # print(f"dataset contains {row['objectId']}")
            dataset.update(row['objectId'], hdu_img)
        else:
            # print(f"dataset DOES NOT contain {row['objectId']}")
            dataset.append(hdu_img)

    return len(object_rows)

def build_groups(objects, dataset_dict):
    groups = defaultdict(list)
    for row in objects:
        groups[(int(row["tract"]), int(row["patch"]))].append(row)
    return [(t, p, rows, dataset_dict) for (t, p), rows in groups.items()]



import numpy as np
import lsst.geom as geom

def wcs_bounds_radec(skywcs, width: int, height: int):
    """
    Return (ra_min_deg, ra_max_deg, dec_min_deg, dec_max_deg) for an image
    with given width/height in pixels using an LSST SkyWcs.

    Handles RA wrap-around (e.g., near 0/360).
    """
    # LSST pixel coords are (x, y). Use edge pixels.
    corners_pix = [
        geom.Point2D(0, 0),
        geom.Point2D(width - 1, 0),
        geom.Point2D(0, height - 1),
        geom.Point2D(width - 1, height - 1),
    ]

    ras = []
    decs = []
    for p in corners_pix:
        sp = skywcs.pixelToSky(p)  # returns lsst.geom.SpherePoint
        ras.append(sp.getRa().asDegrees())
        decs.append(sp.getDec().asDegrees())

    ras = np.array(ras, dtype=float)
    decs = np.array(decs, dtype=float)

    # Fix RA wrap: if corners straddle 0°, unwrap so min/max make sense
    ras_rad = np.deg2rad(ras)
    ras_unwrapped_deg = np.rad2deg(np.unwrap(ras_rad))

    ra_min = float(ras_unwrapped_deg.min())
    ra_max = float(ras_unwrapped_deg.max())

    # put back into [0, 360)
    ra_min = ra_min % 360.0
    ra_max = ra_max % 360.0

    dec_min = float(decs.min())
    dec_max = float(decs.max())

    return ra_min, ra_max, dec_min, dec_max


from astropy.io import fits

import lsst.geom as geom
from astropy.io import fits

def make_cutout_header3(cutout, ra, dec):
    wcs = cutout.getWcs()

    # Start from LSST's FITS WCS keywords
    hdr = fits.Header(wcs.getFitsMetadata().toDict())
    hdr["WCSAXES"] = 2

    # Choose a nice reference pixel in CUTOUT coordinates
    width, height = cutout.getDimensions()  # (W, H)
    x_ref = (width - 1) / 2.0
    y_ref = (height - 1) / 2.0

    # Find the sky coord at that cutout pixel
    sp = wcs.pixelToSky(geom.Point2D(x_ref, y_ref))
    ra_ref = sp.getRa().asDegrees()
    dec_ref = sp.getDec().asDegrees()

    # FITS uses 1-based pixel coordinates for CRPIX
    hdr["CRPIX1"] = x_ref + 1.0
    hdr["CRPIX2"] = y_ref + 1.0
    hdr["CRVAL1"] = ra
    hdr["CRVAL2"] = dec

    ra_min, ra_max, dec_min, dec_max = wcs_bounds_radec(wcs, width, height)

    ra_min = ra + hdr['CD1_1'] * width / 2
    ra_max = ra - hdr['CD1_1'] * width / 2
    dec_min = dec - hdr['CD2_2'] * height / 2
    dec_max = dec + hdr['CD2_2'] * height / 2

    hdr["min_ra"] = ra_min
    hdr["max_ra"] = ra_max
    hdr["min_dec"] = dec_min
    hdr["max_dec"] = dec_max

    return hdr
