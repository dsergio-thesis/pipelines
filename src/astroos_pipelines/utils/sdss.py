
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