import os
import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from astropy.io import fits


from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.utils.plots.as_image import plot_random_samples_as_image
from astroos_pipelines.utils.plots.as_html import plot_random_samples_as_html
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_image'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_html'])

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



def main():


    query_coords = self.pipeline.metadata.get('query_coords')
    query_radius = self.pipeline.metadata.get('query_radius')

    client = AstroosQueryLSST(root_dir=self.stage_dir, 
                              credentials_file=self.pipeline.credentials_file,
                              max_records=self.pipeline.max_records)

    dec_min = max(self.pipeline.metadata.get('query_coords').dec.deg - self.pipeline.metadata.get('query_radius').to(u.deg).value, -90)
    dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

    delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
    ra_min = (query_coords.ra.deg - delta_ra) % 360
    ra_max = (query_coords.ra.deg + delta_ra) % 360


    # Extended Chandra Deep Field South (ECDFS)
    query = \
    """
    SELECT TOP {max_records}
    objectId,
    tract,
    patch,
    coord_ra,
    coord_dec,

    detect_fromBlend, detect_isIsolated,

    -- u
    u_psfFlux,            u_psfFluxErr,            u_psfFlux_flag,
    u_free_cModelFlux,    u_free_cModelFluxErr,    u_free_cModelFlux_flag,

    -- g
    g_psfFlux,            g_psfFluxErr,            g_psfFlux_flag,
    g_free_cModelFlux,    g_free_cModelFluxErr,    g_free_cModelFlux_flag,

    -- r
    r_psfFlux,            r_psfFluxErr,            r_psfFlux_flag,
    r_free_cModelFlux,    r_free_cModelFluxErr,    r_free_cModelFlux_flag,

    -- i
    i_psfFlux,            i_psfFluxErr,            i_psfFlux_flag,
    i_free_cModelFlux,    i_free_cModelFluxErr,    i_free_cModelFlux_flag,

    -- z
    z_psfFlux,            z_psfFluxErr,            z_psfFlux_flag,
    z_free_cModelFlux,    z_free_cModelFluxErr,    z_free_cModelFlux_flag,

    -- y
    y_psfFlux,            y_psfFluxErr,            y_psfFlux_flag,
    y_free_cModelFlux,    y_free_cModelFluxErr,    y_free_cModelFlux_flag,

    refExtendedness

    FROM dp1.Object
    -- WHERE coord_ra BETWEEN 52 AND 54
    --   AND coord_dec BETWEEN -28 AND -26

    -- WHERE coord_ra BETWEEN 53.1 AND 53.2
    --  AND coord_dec BETWEEN -27.9 AND -27.6
    WHERE coord_ra BETWEEN {ra_min} AND {ra_max}
        AND coord_dec BETWEEN {dec_min} AND {dec_max}

    -- AND objectId = 611255072642319851
    """

    query = query.format(
            max_records=self.pipeline.max_records,
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max
            )

    # print(f"Query: {query}")
    
    # sync
    table = client.query(query)

    col_names = table.colnames
    col_types = [str(table[name].dtype) for name in col_names]
    print("Query result columns and types:")

if __name__ == "__main__":
    main()
