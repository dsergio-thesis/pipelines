
from clients._local import *

def main():

    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records
    print("Configuration loaded successfully.")

    transformPolar = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        transforms.CenterCrop(80),
        CropAroundCentroid(crop_size=(100, 100)),
        SegmentationTransform(nsigma=0.5, min_area=40),
        PolarTransform(output_size=(20, 20)),
        # CropZeros(),
    ])
    transformCartesian = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        # transforms.CenterCrop(80),
        # CropAroundCentroid(crop_size=(100, 100)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
        # CropAroundCentroid(crop_size=(30, 30)),
        # CropAroundCentroid(crop_size=(20, 20)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
    ])
    
    dataset_cart_sdss = FITS_Image_Morphometry_Photometry_Dataset(
        dataset_dir=os.path.join(dataset_dir, dataset_name),
        labels_init_file=label_def_file,
        N_bands=5, 
        N_photometric_features=4,
        transform=transformCartesian,
        photometric_transform=MorphometryFeatures()
    )

    print()
    print(dataset_cart_sdss)

    sdss_pipeline = PipelineClassification(
        name=pipeline_name,
        metadata=pipeline_metadata,
        max_records=max_records,
        dataset=dataset_cart_sdss,
        minor_version=None,
    )

    print()
    print(sdss_pipeline)

    pipelines = [
        PipelineDummy(name="dummy_pipeline"),
        sdss_pipeline,
    ]

    pipelines[0].add_stages([
    ])
    
    pipelines[1].add_stages([
        StageCatalogSDSS_V2(),
        StageFetchSDSS_V3_ManualCutout(dataset_cart_sdss),
    ])

    pipelines[1].run_pipeline()

    # for p in pipelines:
    #     p.run_pipeline()


if __name__ == "__main__":
    main()
