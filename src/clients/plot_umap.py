
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
        batch_size=64,
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

    import numpy as np
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)

    import umap

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,   # local vs global structure
        min_dist=0.05,     # cluster tightness
        metric="euclidean",
        random_state=0
    )

    X_umap = reducer.fit_transform(X_scaled)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,6))

    import numpy as np

    colors = np.array(["#1f77b4", "#d62728"])  # blue, red


    # scatter (example)
    plt.scatter(
        X_umap[:, 0],
        X_umap[:, 1],
        c=[colors[i] for i in y],
        s=3,
        alpha=0.3,
        linewidths=0
    )

    plt.xticks([])
    plt.yticks([])
    plt.xlabel("UMAP 1", fontsize=12)
    plt.ylabel("UMAP 2", fontsize=12)
    plt.title("UMAP Projection", fontsize=14)

    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D([0], [0], marker='o', color='w',
               label='Star-forming',
               markerfacecolor="#1f77b4", markersize=8),
        Line2D([0], [0], marker='o', color='w',
               label='Quiescent',
               markerfacecolor="#d62728", markersize=8),
    ]

    plt.legend(
        handles=legend_elements,
        frameon=True,
        facecolor="#eeeeee",
        edgecolor="none",
        fontsize=10,
        loc="upper right"
    )

    # optional polish
    plt.gca().set_aspect('equal', adjustable='box')
    plt.margins(0.05)

    plt.tight_layout()
    plt.savefig("umap.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
