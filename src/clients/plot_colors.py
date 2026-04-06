
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    max_records = config.max_records
    print("Configuration loaded successfully.")
    
    transformCartesian = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        # transforms.CenterCrop(50),
        # CropAroundCentroid(crop_size=(50, 50)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
        # CropAroundCentroid(crop_size=(30, 30)),
        # CropAroundCentroid(crop_size=(20, 20)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
    ])

    random_seed = 1
    random_seed = None
    cmap = 'gist_ncar'
    cmap = cmc.batlow

    dataset_cartesian = FITS_Image_Morphometry_Photometry_Dataset(
        dataset_dir=os.path.join(dataset_dir, dataset_name),
        labels_init_file=os.path.join(dataset_dir, dataset_name, "labels.csv"),
        transform=transformCartesian,
        morphometric_transform=MorphometryFeatures()
    )

    print()
    print(dataset_cartesian)

    labels = dataset_cartesian.get_labels()
    print()
    print("Labels:")
    print(labels)

    val_loader = DataLoaderFITS(
        dataset_cartesian,
        batch_size=512,
        shuffle=False,
        num_workers=0,
    )

    import numpy as np

    def dataloader_to_numpy(loader, max_records=None):
        X_list, y_list = [], []

        if max_records is not None:
            total_records = 0
            for batch in loader:
                _, y, _, X, _ = batch

                print(f"Processing batch with {len(y)} records. Total so far: {total_records}/{max_records}")
                
                X_list.append(X.numpy())
                y_list.append(y.numpy())
                
                total_records += len(y)
                if total_records >= max_records:
                    break
        else:
            for batch in loader:
                _, y, _, X, _ = batch
                
                X_list.append(X.numpy())
                y_list.append(y.numpy())
        return np.concatenate(X_list), np.concatenate(y_list)

    X, y = dataloader_to_numpy(val_loader, max_records=max_records)
    

    # from sklearn.impute import SimpleImputer
    # from sklearn.preprocessing import StandardScaler

    # imputer = SimpleImputer(strategy="median")
    # X_imputed = imputer.fit_transform(X)

    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(X_imputed)

    import matplotlib.pyplot as plt
    import numpy as np
    
    import numpy as np

    import numpy as np

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

    ug = X[:,18]
    gr = X[:,23]
    ri = X[:,27]
    iz = X[:,30]
    zy = X[:,32]

    global_lim = compute_global_limits([ug, gr, ri, iz, zy])

    def plot_color_color(x, y, labels, xlabel, ylabel, filename, xlim, ylim):
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        colors = {
            0: "#1f77b4",
            1: "#d62728",
        }

        mask = np.isfinite(x) & np.isfinite(y)
        x, y, labels = x[mask], y[mask], labels[mask]

        plt.figure(figsize=(4,4))

        for cls in [0, 1]:
            idx = labels == cls
            plt.scatter(
                x[idx], y[idx],
                s=3,
                alpha=0.3,
                c=colors[cls],
                linewidths=0
            )

        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xlabel(f"{xlabel} (blue → red)", fontsize=12)
        plt.ylabel(f"{ylabel} (blue → red)", fontsize=12)

        # 🔥 custom legend
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
        plt.savefig(filename, bbox_inches="tight")
        plt.close()

    print(f"Global limits for color-color plots: {global_lim}")
    print("Plotting color-color diagrams...")

    plot_color_color(
        x=ug, y=gr,
        labels=y,
        xlabel="u-g", ylabel="g-r",
        filename="color_color_ug_gr.png",
        xlim=global_lim, ylim=global_lim
    )

    plot_color_color(
        x=gr, y=ri,
        labels=y,
        xlabel="g-r", ylabel="r-i",
        filename="color_color_gr_ri.png",
        xlim=global_lim, ylim=global_lim
    )

    plot_color_color(
        x=iz, y=zy,
        labels=y,
        xlabel="i-z", ylabel="z-y",
        filename="color_color_iz_zy.png",
        xlim=global_lim, ylim=global_lim
    )

if __name__ == "__main__":
    main()
