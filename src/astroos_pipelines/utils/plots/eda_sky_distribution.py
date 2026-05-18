
import matplotlib.gridspec as gridspec
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import astropy.table
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from collections import defaultdict

def eda_sky_distribution(table: astropy.table.Table, 
                columns: dict, 
                save_dir: str,
                title: str = None, 
                sky_regions: dict = None,
                sample_size: int = None,
                ):
    """
    Perform sky distribution analysis on the given dataset.

    Parameters:
    - table: An astropy Table containing the dataset.
    - columns: A dictionary where keys are column names and values are their descriptions.
    - save_dir: Path to save the generated plots.
    - title: Optional title for the EDA plots.
    - sky_regions: Optional dictionary defining known sky regions with their RA/Dec boundaries. 
                   Format: { "Region Name": (ra_min, ra_max, dec_min, dec_max) }
    - max_records: Maximum number of records to use for plotting (for large datasets).
    """

    os.makedirs(save_dir, exist_ok=True)

    sky_regions = sky_regions or {
        "Galactic Center": (350, 10, -10, 10),
        "CDF-S": (50, 55, -30, -20),
        "COSMOS": (149, 151, 1, 3),
        }
    summary_stats = {}

    # save full table to csv for reference
    # table_file = f"{save_dir}/full_table.csv"
    # table.write(table_file, format='csv', overwrite=True)
    # print(f"Saved full table to {table_file}")

    if (title is None):
        title = "Exploratory Data Analysis"
    
    print(f"\n{title}\nPlotting sky distribution for columns: {columns}")

    print(f"Total records in table: {len(table)}.")

    df = table.to_pandas()
    
    if (sample_size is not None and sample_size < len(df)):
        print(f"Sampling {sample_size} records for plotting (out of {len(df)})...")
        df = df.sample(sample_size, random_state=42)

    if ('ra' in columns and 'dec' in columns):
        ra_col = 'ra'
        dec_col = 'dec'
    elif ('coord_ra' in columns and 'coord_dec' in columns):
        ra_col = 'coord_ra'
        dec_col = 'coord_dec'
    else:
        print("No RA/Dec columns found for sky distribution plot. Skipping sky distribution analysis.")
        ra_col = None
        dec_col = None
        return

    print(f"RA column: {ra_col}, Dec column: {dec_col}")


    if (ra_col in df.columns and dec_col in df.columns): 
        # Build binned dataframe
        plot_df = df[[ra_col, dec_col]].dropna().copy()

        bin_size = 5  # degrees
        plot_df["ra_bin"] = (plot_df[ra_col] // bin_size).astype(int) * bin_size + bin_size / 2
        plot_df["dec_bin"] = (plot_df[dec_col] // bin_size).astype(int) * bin_size + bin_size / 2

        bin_df = (
            plot_df.groupby(["ra_bin", "dec_bin"])
            .size()
            .reset_index(name="count")
        )

        # Keep only bins with enough points
        bin_df = bin_df[bin_df["count"] > 10].copy()

        # Cluster occupied bins
        bin_coords = bin_df[["ra_bin", "dec_bin"]].to_numpy()

        # Since bins are in degrees, eps should also be in degrees.
        # Use something larger than bin_size so adjacent bins connect.
        dbscan = DBSCAN(
            eps=bin_size * 1.5,
            min_samples=1,
            metric="euclidean"
        )

        bin_df["cluster"] = dbscan.fit_predict(bin_coords)

        # Assign cluster labels back to original points
        plot_df = plot_df.merge(
            bin_df[["ra_bin", "dec_bin", "cluster"]],
            on=["ra_bin", "dec_bin"],
            how="left"
        )

        plot_df = plot_df.dropna(subset=["cluster"]).copy()
        plot_df["cluster"] = plot_df["cluster"].astype(int)

        clusters = sorted(plot_df["cluster"].unique())
        n_clusters = len(clusters)

        print(f"DBSCAN found {n_clusters} clusters")

        # Layout: one overall plot + one plot per cluster
        n_plots = n_clusters + 1
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(5 * n_cols, 4 * n_rows),
            constrained_layout=True
        )

        axes = np.array(axes).reshape(-1)

        fig.suptitle(title, fontsize=20)

        # Overall plot
        ax = axes[0]
        ax.scatter(plot_df[ra_col], plot_df[dec_col], s=2, alpha=0.4)
        ax.set_title(f"Overall Sky Distribution ({len(plot_df)} objects)")
        ax.set_xlabel(columns.get(ra_col, ra_col))
        ax.set_ylabel(columns.get(dec_col, dec_col))
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        ax.grid(True, linestyle="--", alpha=0.4)

        # Cluster zoom plots
        for plot_i, cluster_id in enumerate(clusters, start=1):
            ax = axes[plot_i]
            cluster_df = plot_df[plot_df["cluster"] == cluster_id]

            ax.scatter(cluster_df[ra_col], cluster_df[dec_col], s=4, alpha=0.5)

            ra_min, ra_max = cluster_df[ra_col].min(), cluster_df[ra_col].max()
            dec_min, dec_max = cluster_df[dec_col].min(), cluster_df[dec_col].max()

            ra_pad = max((ra_max - ra_min) * 0.1, 0.05)
            dec_pad = max((dec_max - dec_min) * 0.1, 0.05)

            ax.set_xlim(ra_min - ra_pad, ra_max + ra_pad)
            ax.set_ylim(dec_min - dec_pad, dec_max + dec_pad)

            ax.set_title(f"Cluster {cluster_id} ({len(cluster_df)} objects)")
            ax.set_xlabel(columns.get(ra_col, ra_col))
            ax.set_ylabel(columns.get(dec_col, dec_col))
            ax.grid(True, linestyle="--", alpha=0.4)

        # Remove unused axes
        for j in range(n_plots, len(axes)):
            fig.delaxes(axes[j])

        file_name = f"{save_dir}/sky_distribution_clusters.png"
        fig.savefig(file_name, dpi=200)
        print(f"Saved sky distribution cluster plot to {file_name}")
        plt.close(fig)


        # X = df[[ra_col, dec_col]].dropna().values

        # # kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        # # labels = kmeans.fit_predict(X)
        # # centers = kmeans.cluster_centers_
        # coords_rad = np.radians(np.column_stack((X[:, 0], X[:, 1])))
        # eps_arcsec = 60.0 * 3
        # eps_rad = np.deg2rad(eps_arcsec / 3600.0)

        # bin_size = 5  # degrees

        # bins = defaultdict(list)
        # for (ra, dec) in X:
            # # set based on bin_size
            # ra_bin = int(ra // bin_size) * bin_size + bin_size / 2  # center of the bin
            # dec_bin = int(dec // bin_size) * bin_size + bin_size / 2  # center of the bin
            # bins[(ra_bin, dec_bin)].append((ra, dec))

        # print(f"Bins: {bins.keys()}")
        
        # binned_points = [] 
        # for bin_key, points in bins.items():
            # print(f"Bin {bin_key} has {len(points)} points")
            # if len(points) > 10:
                # points = np.array(points)
                # coords_rad = np.radians(points)
                # mean = coords_rad.mean(axis=0)
                # binned_points.append(bin_key)
        # # coords_binned = np.array(binned_points)
        # binned_points = np.radians(np.array(binned_points))
        # print(f"Binned coordinates into {binned_points}")
            

        # dbscan = DBSCAN(
            # eps=eps_rad,
            # min_samples=1,
            # metric='haversine',
            # algorithm='ball_tree',
            # n_jobs=-1,
        # )
        # labels = dbscan.fit_predict(binned_points)

        # unique_labels = set(labels)
        # n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        # print(f"DBSCAN found {n_clusters} clusters")
        # centers = []
        # for label in unique_labels:
            # if label == -1:
                # continue  # skip noise
            # cluster_points = binned_points[labels == label]
            # center = cluster_points.mean(axis=0)
            # centers.append(center)
        # centers = np.array(centers)

        # plt.figure(figsize=(8, 6))
        # # plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)


        # # plot distributions of all clusters
        # n_cols = 3  # number of subplot columns
        # n_rows = int(np.ceil((n_clusters) / n_cols))

        # fig = plt.figure(constrained_layout=True, figsize=(5 * n_cols, 4 * n_rows))
        # fig.suptitle(title, fontsize=24)
        
        # gs = gridspec.GridSpec(
            # n_rows,
            # n_cols,
            # figure=fig,
            # width_ratios=[1.0] * n_cols,
            # height_ratios=[1.0] * n_rows,
        # )

        # for i in range(len(centers)):
            # cluster_points = np.degrees(binned_points[labels == i])
            # center = np.degrees(centers[i])

            # # radius = farthest point in this cluster from center
            # distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            # radius = max(10, distances.max() * 3)

            # circle = plt.Circle((center[0], center[1]), radius, fill=False, linewidth=1, edgecolor='red', alpha=0.5)
            # plt.gca().add_patch(circle)

            # region_name = f"Cluster {i}"
            # if sky_regions is not None:
                # for name, (ra_min, ra_max, dec_min, dec_max) in sky_regions.items():
                    # # print(f"Checking if cluster center {center} is in region '{name}' with RA [{ra_min}, {ra_max}] and Dec [{dec_min}, {dec_max}]")
                    # if ra_min <= center[0] <= ra_max and dec_min <= center[1] <= dec_max:
                        # region_name = name + "\u2020"
                        # break

            # plt.text(center[0] + 20, center[1] + 10, region_name,
                     # ha='center', va='center',
                     # bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))


            # r = i // n_cols
            # c = i % n_cols
            # cluster_df = df[labels == i]
            # ax = fig.add_subplot(gs[r, c])
            # ax.set_title(f"Cluster {i} Distribution")
            # ax.set_xlabel(columns[ra_col])
            # ax.set_ylabel(columns[dec_col])
            # ax.scatter(cluster_df[ra_col], cluster_df[dec_col], s=10, alpha=0.5)


        # sns.scatterplot(x=df[ra_col], y=df[dec_col], s=10, alpha=0.5)
        # # set limits to show the whole sky        
        # plt.xlim(0, 360)
        # plt.ylim(-90, 90)
        # # grid lines every bin_size degrees
        # # plt.xticks(np.arange(0, 361, bin_size))
        # # plt.yticks(np.arange(-90, 91, bin_size))
        # plt.grid(True, linestyle='--', alpha=0.5)

        # plt.title(f"{title} Sky Dist. {len(df)} Objects")
        # plt.xlabel(columns[ra_col])
        # plt.ylabel(columns[dec_col])
        # file_name = f"{save_dir}/sky_distribution.png"
        # plt.savefig(file_name)
        # print(f"Saved sky distribution plot to {file_name}")
        # plt.close()




    for col, desc in columns.items():
        if col in table.colnames and col in [ra_col, dec_col]:
            data = table[col]
            try:
                # convert column to numpy array and drop inf values
                data = np.array(data)
                data = data[~np.isinf(data)]
                summary_stats[col] = {
                    'total': str(len(data)),
                    'n_missing': str(np.sum(np.isnan(data))),
                    'n_inf': str(np.sum(np.isinf(data))),
                    'n_valid': str(np.sum(~np.isnan(data) & ~np.isinf(data))),
                    'mean': str(np.mean(data)), 
                    'median': str(np.median(data)), 
                    'std': str(np.std(data)), 
                    'min': str(data.min()), 
                    'max': str(data.max()) 
                }
            except Exception as e:
                print(f"Could not compute summary statistics for column '{col}': {e}")
                summary_stats[col] = {
                    'mean': None,
                    'median': None,
                    'std': None,
                    'min': None,
                    'max': None
                }
    

    with open(f"{save_dir}/summary_stats.yaml", 'w') as f:
        yaml.dump(summary_stats, f)
    with open (f"{save_dir}/table_info.txt", 'w') as f:
        f.write(str(table.info))


    # remove empty axes if grid > number of columns
    # for j in range(len(columns), n_rows * n_cols):
        # fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

    # file_name = title.lower().replace(" ", "_")
    # file_name = f"{save_dir}/{file_name}.png" 
    # plt.savefig(file_name)
    # print(f"Saved EDA plot to {file_name}")
    # plt.close()

    

