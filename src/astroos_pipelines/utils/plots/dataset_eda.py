
import matplotlib.gridspec as gridspec
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import astropy.table
import pandas as pd
import os


def dataset_eda(table: astropy.table.Table, 
                columns: dict, 
                save_dir: str,
                title: str = None, 
                ):
    """
    Perform exploratory data analysis on the given dataset.

    Parameters:
    - table: An astropy Table containing the dataset.
    - columns: A dictionary where keys are column names and values are their descriptions.
    - save_dir: Path to save the generated plots.
    - title: Optional title for the EDA plots.

    """
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
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=df[ra_col], y=df[dec_col], s=10, alpha=0.5)
        plt.title(f"{title} Sky Distribution (RA vs Dec)")
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
    
    print("\nSummary Statistics:")
    for col, stats in summary_stats.items():
        print(f"\n{col} ({columns[col]}):")
        for stat_name, stat_value in stats.items():
            print(f"  {stat_name.capitalize()}: {stat_value}")

    with open(f"{save_dir}/summary_stats.yaml", 'w') as f:
        yaml.dump(summary_stats, f)
    with open (f"{save_dir}/table_info.txt", 'w') as f:
        f.write(str(table.info))
    


