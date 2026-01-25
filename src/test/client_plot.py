import sys
from torchvision import transforms
import cmcrameri.cm as cmc

if (sys.modules.get('src.astroos.utils.utils') is not None):
    del sys.modules['src.astroos.utils.utils']
from src.astroos.utils.utils import plot_random_samples_from_dataset

if (sys.modules.get('src.astroos.datasets') is not None):
    del sys.modules['src.astroos.datasets']
from src.astroos.datasets import FITS_Image_Features_Dataset

if (sys.modules.get('src.astroos.transforms') is not None):
    del sys.modules['src.astroos.transforms']

from src.astroos.transforms import AddGaussianNoise, \
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
    CropAroundCentroid(crop_size=(30, 30)),
    # CropAroundCentroid(crop_size=(20, 20)),
    # SegmentationTransform(nsigma=0.2, min_area=40),
])

random_seed = 1

if len(sys.argv) > 1:
    num = int(sys.argv[1])
else:
    num = 5
cmap = 'gist_ncar'
cmap = cmc.batlow

dataset_cartesian = FITS_Image_Features_Dataset(
    dir="data/lsst-1",
    N_bands=2, 
    N_features=4, 
    transform=transformCartesian,
    photometric_transform=MorphometryFeatures()
)

plot_random_samples_from_dataset(
    dataset_cartesian,
    num_samples_to_display=num,
    seed=random_seed,
    label_definitions=dataset_cartesian.labels.get_labels(),
    cmap=cmap,
    plot_title="Multi-band Image Cutouts",
    plot_filename="demo_cartesian_transform.png",
    simple_plot=False,
)
