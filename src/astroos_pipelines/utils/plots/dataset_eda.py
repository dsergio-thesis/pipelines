
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

def dataset_eda(table: astropy.table.Table, 
                columns: dict, 
                save_dir: str,
                title: str = None, 
                sky_regions: dict = None
                ):
    """
    Perform exploratory data analysis on the given dataset.

    Parameters:
    - table: An astropy Table containing the dataset.
    - columns: A dictionary where keys are column names and values are their descriptions.
    - save_dir: Path to save the generated plots.
    - title: Optional title for the EDA plots.

    """

    os.makedirs(save_dir, exist_ok=True)

    sky_regions = sky_regions or {
        "Galactic Center": (350, 10, -10, 10),
        "CDF-S": (50, 55, -30, -20),
        "COSMOS": (149, 151, 1, 3),
        }
    summary_stats = {}

    if (title is None):
        title = "Exploratory Data Analysis"
    
    print(f"\n{title}\nPlotting distributions for columns: {columns}")

    df = table.to_pandas()

    if ('ra' in columns and 'dec' in columns):
        ra_col = 'ra'
        dec_col = 'dec'
    elif ('coord_ra' in columns and 'coord_dec' in columns):
        ra_col = 'coord_ra'
        dec_col = 'coord_dec'


    if (ra_col in df.columns and dec_col in df.columns): 
        X = df[[ra_col, dec_col]].dropna().values

        # kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
        # labels = kmeans.fit_predict(X)
        # centers = kmeans.cluster_centers_
        coords_rad = np.radians(np.column_stack((X[:, 0], X[:, 1])))
        eps_arcsec = 60.0 * 3
        eps_rad = np.deg2rad(eps_arcsec / 3600.0)

        bin_size = 5  # degrees

        bins = defaultdict(list)
        for (ra, dec) in X:
            # set based on bin_size
            ra_bin = int(ra // bin_size) * bin_size + bin_size / 2  # center of the bin
            dec_bin = int(dec // bin_size) * bin_size + bin_size / 2  # center of the bin
            bins[(ra_bin, dec_bin)].append((ra, dec))

        print(f"Bins: {bins.keys()}")
        
        binned_points = [] 
        for bin_key, points in bins.items():
            print(f"Bin {bin_key} has {len(points)} points")
            if len(points) > 10:
                points = np.array(points)
                coords_rad = np.radians(points)
                mean = coords_rad.mean(axis=0)
                binned_points.append(bin_key)
        # coords_binned = np.array(binned_points)
        binned_points = np.radians(np.array(binned_points))
        print(f"Binned coordinates into {binned_points}")
            

        dbscan = DBSCAN(
            eps=eps_rad,
            min_samples=1,
            metric='haversine',
            algorithm='ball_tree',
            n_jobs=-1,
        )
        labels = dbscan.fit_predict(binned_points)

        unique_labels = set(labels)
        n_clusters = len(unique_labels) - (1 if -1 in labels else 0)
        print(f"DBSCAN found {n_clusters} clusters")
        centers = []
        for label in unique_labels:
            if label == -1:
                continue  # skip noise
            cluster_points = binned_points[labels == label]
            center = cluster_points.mean(axis=0)
            centers.append(center)
        centers = np.array(centers)

        plt.figure(figsize=(8, 6))
        # plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200)
        for i in range(len(centers)):
            cluster_points = np.degrees(binned_points[labels == i])
            center = np.degrees(centers[i])

            # radius = farthest point in this cluster from center
            distances = np.sqrt(np.sum((cluster_points - center) ** 2, axis=1))
            radius = max(10, distances.max() * 3)

            # circle = plt.Circle((center[0], center[1]), radius, fill=False, linewidth=1, edgecolor='red', alpha=0.5)
            # plt.gca().add_patch(circle)

            region_name = f"Cluster {i}"
            if sky_regions is not None:
                for name, (ra_min, ra_max, dec_min, dec_max) in sky_regions.items():
                    # print(f"Checking if cluster center {center} is in region '{name}' with RA [{ra_min}, {ra_max}] and Dec [{dec_min}, {dec_max}]")
                    if ra_min <= center[0] <= ra_max and dec_min <= center[1] <= dec_max:
                        region_name = name + "\u2020"
                        break

            plt.text(center[0] + 20, center[1] + 10, region_name,
                     ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))


        sns.scatterplot(x=df[ra_col], y=df[dec_col], s=10, alpha=0.5)
        # set limits to show the whole sky        
        plt.xlim(0, 360)
        plt.ylim(-90, 90)
        # grid lines every bin_size degrees
        # plt.xticks(np.arange(0, 361, bin_size))
        # plt.yticks(np.arange(-90, 91, bin_size))
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.title(f"{title} Sky Dist. {len(df)} Objects")
        plt.xlabel(columns[ra_col])
        plt.ylabel(columns[dec_col])
        file_name = f"{save_dir}/sky_distribution.png"
        plt.savefig(file_name)
        print(f"Saved sky distribution plot to {file_name}")
        plt.close()

    n_cols = 3  # number of subplot columns
    n_rows = int(np.ceil(len(columns) / n_cols))

    fig = plt.figure(constrained_layout=True, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle(title, fontsize=24)

    
    gs = gridspec.GridSpec(
        n_rows,
        n_cols,
        figure=fig,
        width_ratios=[1.0] * n_cols,
        height_ratios=[1.0] * n_rows,
    )

    for i, col in enumerate(columns):
        r = i // n_cols
        c = i % n_cols

        ax = fig.add_subplot(gs[r, c])

        data = df[col].dropna()

        # drop inf values
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        sns.histplot(data, bins=50, kde=True, ax=ax)

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(columns[col])
        ax.set_ylabel("Count")

    # remove empty axes if grid > number of columns
    for j in range(len(columns), n_rows * n_cols):
        fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

    file_name = title.lower().replace(" ", "_")
    file_name = f"{save_dir}/{file_name}.png" 
    plt.savefig(file_name)
    print(f"Saved EDA plot to {file_name}")
    plt.close()

    for col, desc in columns.items():
        if col in table.colnames:
            data = table[col]
            try:
                # convert column to numpy array and drop inf values
                data = np.array(data)
                data = data[~np.isinf(data)]
                summary_stats[col] = {
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
    
    # print("\nSummary Statistics:")
    # for col, stats in summary_stats.items():
        # print(f"\n{col} ({columns[col]}):")
        # for stat_name, stat_value in stats.items():
            # print(f"  {stat_name.capitalize()}: {stat_value}")

    with open(f"{save_dir}/summary_stats.yaml", 'w') as f:
        yaml.dump(summary_stats, f)
    with open (f"{save_dir}/table_info.txt", 'w') as f:
        f.write(str(table.info))
    

