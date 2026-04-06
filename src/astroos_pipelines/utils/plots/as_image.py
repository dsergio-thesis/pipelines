import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.ticker as mticker

# Batlow (colorblind-friendly)
import cmcrameri.cm as cmc

def plot_random_samples_as_image(
        dataset, 
        label_definitions,
        num_samples_to_display=None, 
        seed=None,
        cmap='gist_ncar',
        plot_title="",
        plot_show=False,
        predictions=None,
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
        Number of random samples to display (default is 5 or size of dataset).
    seed : int, optional
        Random seed for reproducibility (default is None).
    cmap : str, optional
        Colormap to use for displaying images (default is 'gist_ncar').
    plot_title : str, optional
        Title for the plot (default is empty string).
    plot_filename : str, optional
        If provided, the plot will be saved to this filename instead of displayed (default is None).

    """

    if num_samples_to_display is None:
        num_samples_to_display = min(5, len(dataset))
    else:
        num_samples_to_display = min(num_samples_to_display, len(dataset))
    
    print(f"Plotting {num_samples_to_display} random samples from dataset of size {len(dataset)}")

    rng = np.random.default_rng(seed)
    random_indices = rng.choice(len(dataset), size=num_samples_to_display, replace=False)

    # pull samples once (avoid re-indexing dataset repeatedly)
    samples = [dataset[i] for i in random_indices]
    random_cutouts = [s[0] for s in samples]
    random_labels = [s[1] for s in samples]
    random_morph_features = [s[2] for s in samples]
    random_phot_features = [s[3] for s in samples]
    random_metadata = [s[4] for s in samples]

    print(f"Random phot features: {random_phot_features}")

    # bounds if data exists
    random_image_bounds = None
    if random_cutouts[0] is not None:
        random_image_bounds = [
            (m["MIN_RA"], m["MAX_RA"], m["MIN_DEC"], m["MAX_DEC"]) for m in random_metadata
        ]

    random_samples_info = pd.DataFrame(
        [
            {
                "objectId": str(m.get("OBJECTID", "")),
                "ra": m.get("RA", np.nan),
                "dec": m.get("DEC", np.nan),
                "wcs": m.get("wcs", None),
            }
            for m in random_metadata
        ]
    )
    # summary = []
    # for i in range(num_samples_to_display):
        # summary.append(
            # (random_labels[i], 
             # label_definitions.iloc[int(random_labels[i])]["long_name"] if label_definitions is not None else "",
             # random_samples_info.iloc[i] if not random_samples_info.empty else "",
             # random_morph_features[i] if random_morph_features else ""
            # ))

    bands = ['u', 'g', 'r', 'i', 'z']
    bands = ['u', 'g', 'r', 'i', 'z', 'y']
    num_rows = num_samples_to_display
    num_info_per_row = 2  # info, image, morph features, phot features
    num_cols = len(bands) 

    fig = plt.figure(figsize=(16, num_samples_to_display * num_info_per_row), constrained_layout=True)
    fig.suptitle("LSST DP-1 samples", fontsize=24)
    gs = gridspec.GridSpec(
        num_rows * 2, 
        num_cols, 
        figure=fig,
        width_ratios=[1.0] * num_cols,
        height_ratios=([.25] + [3.0] ) * num_rows,
    )

    for i in range(num_rows):  # rows
        # i += 1
        plot_index = i * 2 - 2 

        label_classname = label_definitions.iloc[int(random_labels[i-1])]["long_name"] if label_definitions is not None else ""
        info = str(random_samples_info.iloc[i-1]['objectId']) if not random_samples_info.empty else ""

        # redshift_info = f"z: {random_samples_info.iloc[i-1]['rvz_redshift']}"
        redshift_info = ""

        ax_info = fig.add_subplot(gs[plot_index, :])
        label_full = f"Label [{label_classname}] objectId [{info}] {redshift_info}"
        label_full = f"{label_classname}"

        ax_info.text(0.5, 0.0, label_full,
                    rotation=0, ha='center', va='center', fontsize=18)
        ax_info.axis('off')

        for j in range(num_cols):       # band column
            print(f"Plotting sample {i-1}, band {bands[j]}, objectId: {random_samples_info.iloc[i-1]['objectId']}, ra: {random_samples_info.iloc[i-1]['ra']}, dec: {random_samples_info.iloc[i-1]['dec']}")

            # plot the cutout
            ax = fig.add_subplot(gs[plot_index + 1, j])
            if random_cutouts[i-1] is not None:
                cutout = random_cutouts[i-1][j]
                extent = random_image_bounds[i-1] if random_image_bounds is not None else None

                wcs = random_metadata[i-1].get("wcs", None)
                print("wcs type:", type(wcs))
                if wcs is not None:
                    # print("pixel_n_dim:", getattr(wcs, "pixel_n_dim", None))
                    # print("world_n_dim:", getattr(wcs, "world_n_dim", None))
                    # if hasattr(wcs, "wcs"):
                        # print("CTYPE:", wcs.wcs.ctype)
                    # pix = wcs.world_to_pixel_values(random_samples_info.iloc[i-1]['ra'], random_samples_info.iloc[i-1]['dec'])
                    # w2 = wcs.celestial  # drops non-celestial axis/axes
                    # w2 = wcs.sub(['longitude', 'latitude'])
                    ra = random_samples_info.iloc[i-1]["ra"]
                    dec = random_samples_info.iloc[i-1]["dec"]
                    ra_min, ra_max, dec_min, dec_max = random_image_bounds[i-1]  # parent-image pixel bounds of the cutout
                    # print(f"random_image_bounds[i-1]: ({xmin}, {xmax}, {ymin}, {ymax})")
                    # x_full, y_full = w2.world_to_pixel_values(ra, dec)
                    # print(f"x_full, y_full: {x_full}, {y_full})")
                    # x_cut = x_full - xmin
                    # y_cut = y_full - ymin
                    # x, y = w2.world_to_pixel_values(
                    # random_samples_info.iloc[i-1]["ra"],
                        # random_samples_info.iloc[i-1]["dec"],
                    # )

                    # corners in world coords
                    # x1, y1 = w2.world_to_pixel_values(ra_min, dec_min)
                    # x2, y2 = w2.world_to_pixel_values(ra_min, dec_max)
                    # x3, y3 = w2.world_to_pixel_values(ra_max, dec_min)
                    # x4, y4 = w2.world_to_pixel_values(ra_max, dec_max)

                    # xs = [x1, x2, x3, x4]
                    # ys = [y1, y2, y3, y4]

                    # x0 = int(np.floor(min(xs)))
                    # y0 = int(np.floor(min(ys)))
# # x1/y1 if you need them:
                    # x1p = int(np.ceil(max(xs)))
                    # y1p = int(np.ceil(max(ys)))
                    # x_full, y_full = w2.world_to_pixel_values(ra, dec)
                    # x = x_full - x0
                    # y = y_full - y0

                    wcs = wcs.sub(['longitude', 'latitude'])  # drop non-celestial axes if present
                    x, y = wcs.world_to_pixel_values(ra, dec)
                    print(f"Calculated pixel coordinates: ({x}, {y}) for RA: {ra}, Dec: {dec} with bounds ({ra_min}, {ra_max}, {dec_min}, {dec_max}) and cutout shape {cutout.shape}")

                    ax.plot(ra, dec, marker='o', color='red', markersize=20, label='Pos', fillstyle='none', markeredgewidth=1) 

                    print(f"Plotted object position at pixel coordinates: ({x}, {y})")
                    # ax.legend(loc='upper right', fontsize=6)

                ax.set_title(f"{bands[j]}", fontsize=14)

                if (j == 0) and (i == 1):
                    ax.set_xlabel("RA °", fontdict={'fontsize': 12}, labelpad=12)
                    ax.set_ylabel("Dec °", fontdict={'fontsize': 12}, labelpad=12)
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=1))
                ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=2))
                ax.tick_params(axis='x', labelsize=12, pad=2)
                ax.tick_params(axis='y', labelsize=12, pad=2)
                ax.imshow(cutout,
                        cmap=cmap,
                        extent=extent if extent is not None else None,
                        aspect='auto'
                        # cmap='plasma',
                        # cmap=cmc.batlow, # better for colorblind
                )
            else:
                ax.set_facecolor("lightgray")
                ax.text(0.5, 0.5, "(no data)",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=12, color="dimgray")
                ax.set_xticks([])
                ax.set_yticks([])

            # plot morphometry features 
            # ax_cash = fig.add_subplot(gs[plot_index + 2, j])
            # if random_morph_features[i-1] is not None:
                # features_morph = random_morph_features[i-1][j]
                # features_morph = np.array(features_morph, dtype=np.float32)
                # x = np.array([0, 1, 2, 3])
                # ax_cash.bar(x, features_morph, width=0.25, color='skyblue', edgecolor='black')
                # ax_cash.set_xticks(x)
                # ax_cash.set_yticks(np.linspace(0.0, 1.0, num=3))
                # ax_cash.set_xticklabels(['C', 'A', 'S', 'H'])
                # ax_cash.tick_params(axis='x', labelsize=14, pad=2)
                # ax_cash.set_ylim(0.0, 1.0)
                # ax_cash.set_xlabel('Morphometry', fontsize=12)
                # ax_cash.set_ylabel('Norm', fontsize=12)
            # else:
                # ax_cash.set_facecolor("lightgray")
                # ax_cash.text(0.5, 0.5, "(no data)",
                # ha="center", va="center",
                # transform=ax_cash.transAxes,
                # fontsize=12, color="dimgray")
                # ax_cash.set_xticks([])
                # ax_cash.set_yticks([])

            # plot photometry features
            # ax_phot = fig.add_subplot(gs[plot_index + 3, j])
            # if random_phot_features[i-1] is not None:
                # features_phot = random_phot_features[i-1][j]
                # features_phot = np.array(features_phot, dtype=np.float32)
                # print(f"Photometry features for sample {i-1}, band {bands[j]}: {features_phot}")
                # # x = np.linspace(0, len(features_phot)-1, num=len(features_phot))
                # x = np.array([0, 1, 2, 3])
                # ax_phot.bar(x, features_phot, color='skyblue', edgecolor='black')
                # ax_phot.set_xticks(x)
                # # ax_phot.set_xticklabels(['x1', 'x2', 'x3', 'x4'])
                # ax_phot.tick_params(axis='x', labelsize=11, pad=2)
                # lower = 0.0 if np.min(features_phot) > 0 else np.min(features_phot) * 2
                # upper = np.max(features_phot) * 2 if np.max(features_phot) > 0 else 0.0
                # # ax_phot.set_ylim(float(lower), float(upper))
                # ax_phot.set_xlabel('Photometry', fontsize=12)
                # ax_phot.set_ylabel('Value', fontsize=12)
            # else:
                # ax_phot.set_facecolor("lightgray")
                # ax_phot.text(0.5, 0.5, "(no data)",
                # ha="center", va="center",
                # transform=ax_phot.transAxes,
                # fontsize=12, color="dimgray")
                # ax_phot.set_xticks([])
                # ax_phot.set_yticks([])
            

    output_dir = os.path.join(dataset.get_dataset_dir(), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    plot_filename = f"{dataset.get_dataset_dir()}/plots/random_samples_{num_samples_to_display}.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved to {plot_filename}")

    if plot_show:
        plt.show()


