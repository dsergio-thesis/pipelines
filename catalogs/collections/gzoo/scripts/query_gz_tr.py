
query["description"] = "Get 10 random objects"
max_records = parameters.get("max_records", 10)
# query["adql"] = """
# SELECT TOP 10 *
# FROM ls_dr10.tractor
# """

query["adql"] = """
SELECT TOP {max_records} 
    gz.ra,
    gz.dec,
    gz.smooth_or_featured_artifact_fraction,
    gz.smooth_or_featured_smooth_fraction,
    gz.smooth_or_featured_featured_or_disk_fraction,
    gz.has_spiral_arms_yes_fraction,
    gz.disk_edge_on_yes_fraction,
    gz.bar_strong_fraction,
    gz.bar_weak_fraction,
    gz.merging_merger_fraction,
    gz.merging_major_disturbance_fraction,
    gz.merging_minor_disturbance_fraction,
    gz.how_rounded_round_fraction,
    gz.how_rounded_cigar_shaped_fraction,


    tr.brickid,
    tr.brickname,
    tr.objid AS objectId,
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

FROM ls_dr8.galaxy_zoo AS gz
JOIN ls_dr8.tractor AS tr ON (tr.ra = gz.ra AND tr.dec = gz.dec)

WHERE gz.ra BETWEEN 190.0 AND 200.0
AND gz.dec BETWEEN 30.0 AND 35.0

"""


query["adql"] = query["adql"].format(max_records=max_records)
