import os
import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from config.astroos_config import AstroosConfig
from utils.plot_utils import plot_random_samples_from_dataset
from astroos_pipelines.datasets import FITS_Image_Morphometry_Photometry_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid

importlib.reload(sys.modules['astroos_pipelines.datasets'])
importlib.reload(sys.modules['astroos_pipelines.transforms'])
importlib.reload(sys.modules['config.astroos_config'])
importlib.reload(sys.modules['utils.plot_utils'])

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
        transform=transformCartesian,
        morphometric_transform=MorphometryFeatures()
    )

    print()
    print(dataset_cartesian)

    plot_random_samples_from_dataset(
        dataset_cartesian, 
        num_samples_to_display=max_records,
        seed=random_seed, 
        label_definitions=dataset_cartesian.get_labels(), 
        cmap=cmap,
        plot_title="LSST Cartesian Samples",
        plot_filename="lsst_cart.png",
        simple_plot=False,
    )

if __name__ == "__main__":
    main()
