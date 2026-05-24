
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
    gz.how_rounded_cigar_shaped_fraction

FROM ls_dr8.galaxy_zoo AS gz

WHERE ra BETWEEN 150.0 AND 250.0 
AND dec BETWEEN 0.0 AND 60.0

"""


query["adql"] = query["adql"].format(max_records=max_records)
