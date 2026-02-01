import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from astroos_pipelines.utils.utils import plot_random_samples_from_dataset
from astroos_pipelines.datasets import FITS_Image_Features_Dataset
from astroos_pipelines.transforms import AddGaussianNoise, \
    MorphometryFeatures, \
    SegmentationTransform, \
    PolarTransform, \
    CropZeros, \
    CropAroundCentroid


transformCartesian = transforms.Compose([
    # transforms.ToTensor(),
    # AddGaussianNoise(mean=0., std=0.3),
    # transforms.CenterCrop(30),
    # CropAroundCentroid(crop_size=(100, 100)),
    # SegmentationTransform(nsigma=0.2, min_area=40),
    # CropAroundCentroid(crop_size=(30, 30)),
    # CropAroundCentroid(crop_size=(20, 20)),
    # SegmentationTransform(nsigma=0.2, min_area=40),
])

random_seed = 1

if len(sys.argv) > 1:
    num = int(sys.argv[1])
else:
    num = 5
cmap = 'gist_ncar'
# cmap = cmc.batlow

dataset_cartesian = FITS_Image_Features_Dataset(
    dir="data/lsst-4",
    N_bands=5, 
    photometric_transform=None
)

print(f"Dataset size: {len(dataset_cartesian)} samples.")

plot_random_samples_from_dataset(
    dataset_cartesian, 
    num_samples_to_display=num,
    seed=random_seed, label_definitions=dataset_cartesian.labels.get_labels(), 
    cmap=cmap,
    plot_title="Multi-band Image Cutouts",
    plot_filename="lsst_cart.png",
    simple_plot=False,
)
