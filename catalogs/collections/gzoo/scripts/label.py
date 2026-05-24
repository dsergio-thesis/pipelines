def galaxy_zoo_label(row):
    artifact = row.smooth_or_featured_artifact_fraction
    smooth = row.smooth_or_featured_smooth_fraction
    disk = row.smooth_or_featured_featured_or_disk_fraction

    spiral_yes = row.has_spiral_arms_yes_fraction
    edge_on = row.disk_edge_on_yes_fraction

    bar_strong = row.bar_strong_fraction
    bar_weak = row.bar_weak_fraction

    merger = row.merging_merger_fraction
    major = row.merging_major_disturbance_fraction
    minor = row.merging_minor_disturbance_fraction

    rounded = row.how_rounded_round_fraction
    cigar = row.how_rounded_cigar_shaped_fraction

    if artifact > 0.5:
        return "Unknown"

    if merger > 0.4 or major > 0.4:
        return "Merger"

    if minor > 0.4:
        return "Pec"

    if smooth > 0.8:
        if cigar > 0.5:
            return "E6"
        if rounded > 0.7:
            return "E0"
        return "E"

    if disk > 0.6:
        if edge_on > 0.7:
            return "S0"

        if spiral_yes > 0.6:
            if bar_strong > 0.4:
                return "SBc"
            if bar_weak > 0.4:
                return "SBbc"
            return "Sc"

        return "S0"

    return "Unknown"


labels = {"Unknown": 0} 
if "labels" in parameters:
    with open(parameters["labels"], "r") as f:
        for line in f:
            line_arr = line.strip().split(",")
            labels[line_arr[1]] = line_arr[0]

for row in df.itertuples():
    label_name = galaxy_zoo_label(row)
    if label_name in labels:
        df.at[row.Index, "label"] = labels[label_name]
    else:
        df.at[row.Index, "label"] = labels["Unknown"]


