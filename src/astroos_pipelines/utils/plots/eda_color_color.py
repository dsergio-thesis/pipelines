
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

def eda_color_color(table: astropy.table.Table, 
                columns: dict, 
                save_dir: str,
                title: str = None, 
                sample_size: int = None,
                ):
    """
    Perform color-color analysis on the given dataset.

    Parameters:
    - table: An astropy Table containing the dataset.
    - columns: A dictionary where keys are column names and values are their descriptions.
    - save_dir: Path to save the generated plots.
    - title: Optional title for the EDA plots.
    - max_records: Maximum number of records to use for plotting (for large datasets).
    """

    os.makedirs(save_dir, exist_ok=True)

    summary_stats = {}

    # save full table to csv for reference
    # table_file = f"{save_dir}/full_table.csv"
    # table.write(table_file, format='csv', overwrite=True)
    # print(f"Saved full table to {table_file}")

    if (title is None):
        title = "Exploratory Data Analysis"
    
    print(f"\n{title}\nPlotting color-color for columns: {columns}")

    print(f"Total records in table: {len(table)}.")

    df = table.to_pandas()
    
    if (sample_size is not None and sample_size < len(df)):
        print(f"Sampling {sample_size} records for plotting (out of {len(df)})...")
        df = df.sample(sample_size, random_state=42)


    def compute_global_limits(arrays, lower=1, upper=99):
        # concatenate all arrays
        all_vals = np.concatenate([np.asarray(a).ravel() for a in arrays])

        # remove NaN / Inf
        all_vals = all_vals[np.isfinite(all_vals)]

        if len(all_vals) == 0:
            return (-1, 1)

        lo, hi = np.percentile(all_vals, [lower, upper])

        if lo == hi:
            lo -= 1e-3
            hi += 1e-3

        return lo, hi


    def plot_color_color(x, y, labels, xlabel, ylabel, filename, xlim, ylim):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        colors = {
            0: "#1f77b4",
            1: "#d62728",
        }

        mask = np.isfinite(x) & np.isfinite(y)

        if labels is not None:
            x, y, labels = x[mask], y[mask], labels[mask]
        else:
            x, y = x[mask], y[mask]


        print(f"x, y, labels shapes after masking: {x.shape}, {y.shape}, {labels.shape if labels is not None else 'N/A'}")

        plt.figure(figsize=(4,4))

        if labels is not None:
            for cls in [0, 1]:
                idx = labels == cls
                plt.scatter(
                    x[idx], y[idx],
                    s=3,
                    alpha=0.3,
                    c=colors[cls],
                    linewidths=0
                )
        else:
            plt.scatter(
                x, y,
                s=3,
                alpha=0.3,
                c=colors[0],
                linewidths=0
            )

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(f"{xlabel} (blue → red)", fontsize=12)
        plt.ylabel(f"{ylabel} (blue → red)", fontsize=12)

        if labels is not None:

            # custom legend
            legend_elements = [
                Line2D([0], [0], marker='o', color='w',
                       label='Star-forming',
                       markerfacecolor=colors[0], markersize=6),
                Line2D([0], [0], marker='o', color='w',
                       label='Quiescent',
                       markerfacecolor=colors[1], markersize=6),
            ]

            plt.legend(
                handles=legend_elements,
                frameon=True,
                facecolor="#eeeeee",
                edgecolor="none",
                fontsize=10
            )

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
        plt.close()


    labels = np.asarray(df['label']) if 'label' in df.columns else None
    ug = np.asarray(df['color_ug']) if 'color_ug' in df.columns else None 
    gr = np.asarray(df['color_gr']) if 'color_gr' in df.columns else None 
    ri = np.asarray(df['color_ri']) if 'color_ri' in df.columns else None 
    iz = np.asarray(df['color_iz']) if 'color_iz' in df.columns else None 
    zy = np.asarray(df['color_zy']) if 'color_zy' in df.columns else None 

    all_arrays = [arr for arr in [ug, gr, ri, iz, zy] if arr is not None]

    global_lim = compute_global_limits(all_arrays, lower=1, upper=99)

    print(f"Global limits for color-color plots: {global_lim}")
    print("Plotting color-color diagrams...")
    print(f"labels shape: {labels.shape if labels is not None else 'N/A'}, ug shape: {ug.shape if ug is not None else 'N/A'}, gr shape: {gr.shape if gr is not None else 'N/A'}, ri shape: {ri.shape if ri is not None else 'N/A'}, iz shape: {iz.shape if iz is not None else 'N/A'}, zy shape: {zy.shape if zy is not None else 'N/A'}")

    plot_color_color(
        x=ug, y=gr,
        labels=None,
        xlabel="u-g", ylabel="g-r",
        filename="color_color_ug_gr.png",
        xlim=global_lim, ylim=global_lim
    )

    plot_color_color(
        x=gr, y=ri,
        labels=None,
        xlabel="g-r", ylabel="r-i",
        filename="color_color_gr_ri.png",
        xlim=global_lim, ylim=global_lim
    )

    plot_color_color(
        x=iz, y=zy,
        labels=None,
        xlabel="i-z", ylabel="z-y",
        filename="color_color_iz_zy.png",
        xlim=global_lim, ylim=global_lim
    )













    for col, desc in columns.items():
        if col in table.colnames:
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


