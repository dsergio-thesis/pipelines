
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

query_coords = parameters.get('query_coords')
query_radius = parameters.get('query_radius')
max_records = parameters.get('max_records')

if query_coords is dict
query_coords = SkyCoord(
        ra=query_coords["ra_deg"] * u.deg,
        dec=query_coords["dec_deg"] * u.deg,
        frame=query_coord["frame"]
        )
query_radius = 


if query_radius > 0:
    dec_min = max(query_coords.dec.deg - query_radius.to(u.deg).value, -90)
    dec_max = min(query_coords.dec.deg + query_radius.to(u.deg).value, 90)

    delta_ra = query_radius.to(u.deg).value / np.cos(np.deg2rad(query_coords.dec.deg))
    ra_min = (query_coords.ra.deg - delta_ra) % 360
    ra_max = (query_coords.ra.deg + delta_ra) % 360


    # Query for objects within the RA/Dec box defined by the center and radius.
    query["adql"] = \
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
    
    WHERE coord_ra BETWEEN {ra_min} AND {ra_max}
        AND coord_dec BETWEEN {dec_min} AND {dec_max}

    """
    query["adql"] = query["adql"].format(
            max_records=max_records,
            ra_min=ra_min,
            ra_max=ra_max,
            dec_min=dec_min,
            dec_max=dec_max
            )
else:

    # Query for all objects (up to max_records limit)
    query["adql"] = \
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
    """

    query["adql"] = query["adql"].format(
            max_records=max_records,
            )
