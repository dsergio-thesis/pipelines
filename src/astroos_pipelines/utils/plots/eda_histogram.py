
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

def eda_histogram(table: astropy.table.Table, 
                columns: dict, 
                save_dir: str,
                title: str = None, 
                sample_size: int = None,
                ):
    """
    Generate histograms on the given dataset.

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
    table_file = f"{save_dir}/full_table.csv"
    table.write(table_file, format='csv', overwrite=True)
    print(f"Saved full table to {table_file}")

    if (title is None):
        title = "Exploratory Data Analysis"
    
    print(f"\n{title}\nPlotting distributions for columns: {columns}")

    print(f"Total records in table: {len(table)}.")

    df = table.to_pandas()
    
    if (sample_size is not None and sample_size < len(df)):
        print(f"Sampling {sample_size} records for plotting (out of {len(df)})...")
        df = df.sample(sample_size, random_state=42)

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
    
    # print("\nSummary Statistics:")
    # for col, stats in summary_stats.items():
        # print(f"\n{col} ({columns[col]}):")
        # for stat_name, stat_value in stats.items():
            # print(f"  {stat_name.capitalize()}: {stat_value}")

    with open(f"{save_dir}/summary_stats.yaml", 'w') as f:
        yaml.dump(summary_stats, f)
    with open (f"{save_dir}/table_info.txt", 'w') as f:
        f.write(str(table.info))

    # this should be in the user-provided script
    # for col in ['sfr', 'sfr_UV', 'sfr_IR']:
        # columns.pop(col, None)

    # remove all c in columns that are not in df.columns
    columns = {c: desc for c, desc in columns.items() if c in df.columns}

    n_cols = 3  # number of subplot columns
    n_rows = int(np.ceil((len(columns)) / n_cols))  

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

        data = pd.to_numeric(df[col], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            ax.set_title(f"Distribution of {col}")
            ax.text(0.5, 0.5, "No numeric data", ha="center", va="center")
            ax.set_axis_off()
            continue


        # print(f"data: {data}")

        # KDE fails if there are too few values or all values are identical
        use_kde = len(data) > 1 and data.nunique() > 1

        try:
            sns.histplot(data, bins=50, kde=True, ax=ax)
        except ValueError:
            sns.histplot(data, bins=50, kde=False, ax=ax) 

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(columns[col] if col in columns else col)
        ax.set_ylabel("Count")

        # ---- Save individual plot ----
        indiv_fig, indiv_ax = plt.subplots(figsize=(6, 4))
        sns.histplot(data, bins=50, kde=True, ax=indiv_ax)

        indiv_ax.set_title(f"Distribution of {col}")
        indiv_ax.set_xlabel(columns[col])
        indiv_ax.set_ylabel("Count")

        indiv_file = f"{save_dir}/{col.lower().replace(' ', '_')}.png"
        indiv_fig.savefig(indiv_file, bbox_inches="tight", dpi=300)
        plt.close(indiv_fig)


    # remove empty axes if grid > number of columns
    for j in range(len(columns), n_rows * n_cols):
        fig.delaxes(fig.add_subplot(gs[j // n_cols, j % n_cols]))

    file_name = title.lower().replace(" ", "_")
    file_name = f"{save_dir}/{file_name}.png" 
    plt.savefig(file_name)
    print(f"Saved EDA plot to {file_name}")
    plt.close()

    

