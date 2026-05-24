
query["description"] = "Get 10 random objects"
max_records = parameters.get("max_records", 10)
# query["adql"] = """
# SELECT TOP 10 *
# FROM ls_dr10.tractor
# """

query["adql"] = """
SELECT TOP {max_records} 
    tr.ra,
    tr.dec,
    tr.brickid,
    tr.brickname,
    tr.objid,
    tr.bx AS brick_x,
    tr."by" AS brick_y, 

    tr.type,
    tr.mag_g,
    tr.mag_r,
    tr.mag_z,
    tr.g_r,
    tr.r_z,
    tr.z_w1,

    tr.flux_g,
    tr.flux_r,
    tr.flux_z,
    tr.flux_w1,
    tr.flux_w2,

    tr.shapeexp_r,
    tr.shapeexp_e1,
    tr.shapeexp_e2,
    tr.shapedev_r,
    tr.shapedev_e1,
    tr.shapedev_e2,
    tr.fracdev,

    tr.snr_g,
    tr.snr_r,
    tr.snr_z

FROM ls_dr8.tractor AS tr

WHERE ra BETWEEN 286.0 AND 287.0 
AND dec BETWEEN 43.5 AND 44.5

"""


query["adql"] = query["adql"].format(max_records=max_records)
