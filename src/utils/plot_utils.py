
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import time

# device = torch.device("cpu") #default device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.ticker as mticker


# Batlow
import cmcrameri.cm as cmc

import warnings
warnings.filterwarnings("error", message="Mean of empty slice")

def plot_random_samples_from_dataset(
        dataset, 
        label_definitions,
        num_samples_to_display=5, 
        seed=None,
        cmap='gist_ncar',
        plot_title="",
        plot_filename=None,
        simple_plot=False,
        ):
    """
    Plot random samples from a dataset with their labels and features.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to sample from. Each item should return (image, label, morph_features, phot_features, metadata).
    label_definitions : pd.DataFrame
        DataFrame containing label definitions with 'short_name' as index and 'long_name' as a column.
    num_samples_to_display : int, optional
        Number of random samples to display (default is 5).
    seed : int, optional
        Random seed for reproducibility (default is None).
    cmap : str, optional
        Colormap to use for displaying images (default is 'gist_ncar').
    plot_title : str, optional
        Title for the plot (default is empty string).
    plot_filename : str, optional
        If provided, the plot will be saved to this filename instead of displayed (default is None).
    simple_plot : bool, optional
        If True, only display the main_id as label. If False, display the full label with class name and redshift (default is False).
    """
    
    print(f"Plotting {num_samples_to_display} random samples from dataset of size {len(dataset)}")
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed()
    random_indices = np.random.choice(len(dataset), size=num_samples_to_display, replace=False)

    print("First index:")
    print(dataset[0])

    # print("Random indices:", random_indices)
    random_samples = [dataset[i][0] for i in random_indices]
    random_labels = [dataset[i][1] for i in random_indices]
    random_morph_features = [dataset[i][2] for i in random_indices]
    random_phot_features = [dataset[i][3] for i in random_indices]
    random_image_bounds = [(dataset[i][4]['MIN_RA'], dataset[i][4]['MAX_RA'],
                            dataset[i][4]['MIN_DEC'], dataset[i][4]['MAX_DEC'])
                           for i in random_indices]
    random_main_ids = [dataset[i][4]['MAIN_ID'] for i in random_indices]
    random_ras = [dataset[i][4]['RA'] for i in random_indices]
    random_decs = [dataset[i][4]['DEC'] for i in random_indices]
    random_rvz = [dataset[i][4]['rvz_redshift'] for i in random_indices]
    for i in range(len(random_samples)):
        print(random_samples[i].shape)
        print("mean: ", random_samples[i].mean())

    # labels = label_definitions("./sdss_morph_types_info.csv")
    # print(labels)
    # print("Random labels:", random_labels)
    # print("Random bounds:", random_image_bounds)
    # print("random samples: ", np.array(random_samples).shape)
    # print("label definitions:", label_definitions)


    # if sample_information_file is not None:
    #     random_samples_info = pd.read_csv(sample_information_file)[['main_id', 'rvz_redshift', 'galdim_majaxis', 'galdim_minaxis', 'galdim_angle']]
    #     # get only the rows corresponding to random_indices
    #     random_samples_info = random_samples_info.iloc[random_indices]

    print("random_image_bounds:", random_image_bounds)

    random_samples_info = pd.DataFrame()

    for i in range(len(random_main_ids)):
        row = pd.DataFrame([{
            'main_id': str(random_main_ids[i]),
            'rvz_redshift': random_rvz[i],
            'ra': random_ras[i],
            'dec': random_decs[i],
        }])
        random_samples_info = pd.concat([random_samples_info, row], ignore_index=True)


    for i in random_samples_info.index:
        if random_samples_info.at[i, 'rvz_redshift'] == '--':
            random_samples_info.at[i, 'rvz_redshift'] = 0.0 # do something else here

    summary = []
    for i in range(num_samples_to_display):
        summary.append(
            (random_labels[i], 
             label_definitions.iloc[int(random_labels[i])]["long_name"],
             random_samples_info.iloc[i] if not random_samples_info.empty else "",
             random_morph_features[i] if random_morph_features else ""
            ))
        
    # print("Random samples summary (label index, sample info):")
    # for item in summary:
    #     print(item)

    # print(random_morph_features)
    # print(random_samples_info)
    # print(label_definitions)
    # print(summary)


    bands = ['u', 'g', 'r', 'i', 'z']
    # bands = ['u', 'g']
    num_rows = num_samples_to_display
    num_cols = len(bands) 

    fig = plt.figure(figsize=(16, num_samples_to_display * 4), constrained_layout=True)
    fig.suptitle(plot_title, fontsize=24)
    gs = gridspec.GridSpec(
        num_rows * 4, 
        num_cols, 
        figure=fig,
        width_ratios=[1.0] * num_cols,
        height_ratios=([.25] + [3.0] + [1.0] + [1.0]) * num_rows,
    )

    for i in range(num_rows):  # rows
        # i += 1
        plot_index = i * 4 - 4

        label_classname = label_definitions.iloc[int(random_labels[i-1])]["long_name"]
        info = str(random_samples_info.iloc[i-1]['main_id']) if not random_samples_info.empty else ""

        redshift = random_samples_info.iloc[i-1]['rvz_redshift']
        ax_info = fig.add_subplot(gs[plot_index, :])

        if simple_plot:
            label_full = f"{info}"
        else:
            label_full = f"{label_classname} {info} (z={redshift})"

        ax_info.text(0.5, 0.0, label_full,
                    rotation=0, ha='center', va='center', fontsize=18)
        ax_info.axis('off')

        

        for j in range(num_cols):       # image columns
            
            sample = random_samples[i-1][j]
            
            extent = random_image_bounds[i-1] if random_image_bounds is not None else None
            print(f"Plotting sample {i-1}, band {bands[j]}, main_id: {random_samples_info.iloc[i-1]['main_id']}, ra: {random_samples_info.iloc[i-1]['ra']}, dec: {random_samples_info.iloc[i-1]['dec']}")
            # print(f"extent.shape: {extent.shape} extent: {extent} sample.shape: {sample.shape}" \
            #       f" ra_diff: {extent[1]-extent[0]} dec_diff: {extent[3]-extent[2]}")
            
            # ra_min, ra_max, dec_min, dec_max = extent
            # ra_center = 0.5*(ra_min + ra_max)
            # ra_diff = dec_max - dec_min  # make RA span = Dec span for plotting
            # extent_plot = [ra_center - ra_diff/2, ra_center + ra_diff/2, dec_min, dec_max]

            ax = fig.add_subplot(gs[plot_index + 1, j])
            ax.imshow(sample,
                    cmap=cmap,
                    extent=extent if extent is not None else None,
                    aspect='auto'

                    # cmap='plasma',
                    # cmap=cmc.batlow, # better for colorblind
            )
            # ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

            ax.set_title(f"band: {bands[j]}", fontsize=14)
            # ax.axis('off')
            ax.set_xlabel("RA °", fontdict={'fontsize': 18}, labelpad=14)
            ax.set_ylabel("Dec °", fontdict={'fontsize': 18}, labelpad=14)
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))

            ax.tick_params(axis='x', labelsize=12, pad=2)
            ax.tick_params(axis='y', labelsize=12, pad=2)

            
            # ax_label.text(0.5, 0.5, f"features: {1}",
            #         rotation=0, ha='center', va='center', fontsize=12)
            ax_cash = fig.add_subplot(gs[plot_index + 2, j])
            features = random_morph_features[i-1][j]
            # features = np.concatenate([features, np.array([random_samples_info.iloc[i-1]['rvz_redshift']])], axis=0)  # for R
            features = np.array(features, dtype=np.float32)
            # print(f"features: {features} features.shape: {features.shape}")
            x = np.array([0, 1, 2, 3])

            # set features to random numbers
            # features = np.random.uniform(0, 1, size=(4)) 

            ax_cash.bar(x, features, width=0.25, color='skyblue', edgecolor='black')
            # ax_cash.scatter(x, features, marker='o', color='blue')
            ax_cash.set_xticks(x)
            ax_cash.set_yticks(np.linspace(0.0, 1.0, num=3))
            ax_cash.set_xticklabels(['C', 'A', 'S', 'H'])
            ax_cash.tick_params(axis='x', labelsize=14, pad=2)
            ax_cash.set_ylim(0.0, 1.0)
            ax_cash.set_xlabel('Morphometry', fontsize=12)
            ax_cash.set_ylabel('Norm', fontsize=12)



            ax_cash = fig.add_subplot(gs[plot_index + 3, j])
            phot = random_phot_features[i-1][j]
            phot = np.array(phot, dtype=np.float32)
            x = np.array([0, 1, 2, 3])
            ax_cash.bar(x, phot, width=0.25, color='skyblue', edgecolor='black')
            ax_cash.set_xticks(x)
            # ax_cash.set_yticks(np.linspace(0.0, 1.0, num=3))
            ax_cash.set_xticklabels(['x1', 'x2', 'x3', 'x4'])
            ax_cash.tick_params(axis='x', labelsize=11, pad=2)
            ax_cash.set_ylim(0.0 if np.min(phot) > 0 else np.min(phot) * 2, np.max(phot) * 2)
            ax_cash.set_xlabel('Photometry', fontsize=12)
            ax_cash.set_ylabel('Value', fontsize=12)


            # ax_cash.set_aspect(2.0)
            # ax_cash.set_adjustable('datalim')
            # ax_cash.axis('off')
            
    # fig.subplots_adjust(left=0.2, right=0.9, top=1.0, bottom=0.0)
    # Add horizontal lines between rows
    # for i in range(1, num_rows * 3):
    #     # compute y in figure coordinates
    #     if (i % 3) == 0:
    #         y = 1 - i/(num_rows * 3)  # adjust if needed
    #         line = mlines.Line2D([0, 1], [y, y], transform=fig.transFigure, color='black', linewidth=1)
    #         fig.lines.append(line)
    if plot_filename:
        plt.savefig(plot_filename, dpi=300)
    else:
        plt.show()
