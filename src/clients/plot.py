
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")
    
    transformCartesian = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        transforms.CenterCrop(50),
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


    plot_random_samples_as_image(
        dataset_cartesian, 
        num_samples_to_display=max_records,
        seed=random_seed, 
        label_definitions=dataset_cartesian.get_labels(), 
        cmap=cmap,
        plot_title="LSST Cartesian Samples",
        plot_show=True,
    )
    # plot_random_samples_as_html(
        # dataset_cartesian, 
        # num_samples_to_display=max_records,
        # seed=random_seed, 
        # label_definitions=dataset_cartesian.get_labels(), 
        # cmap=cmap,
        # plot_title="LSST Cartesian Samples",
    # )

if __name__ == "__main__":
    main()
