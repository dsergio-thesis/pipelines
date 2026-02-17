import os
import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from astroos_pipelines.config.astroos_config import AstroosConfig
from astroos_pipelines.utils.plots.as_image import plot_random_samples_as_image
from astroos_pipelines.utils.plots.as_html import plot_random_samples_as_html
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['astroos_pipelines.config.astroos_config'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_image'])
importlib.reload(sys.modules['astroos_pipelines.utils.plots.as_html'])

def main():

    config = AstroosConfig.from_cli()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    max_records = config.max_records
    print()
    print(config)
    
    transformCartesian = transforms.Compose([
        # transforms.ToTensor(),
        # AddGaussianNoise(mean=0., std=0.3),
        # transforms.CenterCrop(30),
        CropAroundCentroid(crop_size=(30, 30)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
        # CropAroundCentroid(crop_size=(30, 30)),
        # CropAroundCentroid(crop_size=(20, 20)),
        # SegmentationTransform(nsigma=0.2, min_area=40),
    ])

    random_seed = 1
    cmap = 'gist_ncar'
    # cmap = cmc.batlow

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


    # plot_random_samples_as_image(
        # dataset_cartesian, 
        # num_samples_to_display=max_records,
        # seed=random_seed, 
        # label_definitions=dataset_cartesian.get_labels(), 
        # cmap=cmap,
        # plot_title="LSST Cartesian Samples",
        # plot_show=True,
    # )
    plot_random_samples_as_html(
        dataset_cartesian, 
        num_samples_to_display=max_records,
        seed=random_seed, 
        label_definitions=dataset_cartesian.get_labels(), 
        cmap=cmap,
        plot_title="LSST Cartesian Samples",
    )

if __name__ == "__main__":
    main()
