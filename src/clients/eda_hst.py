import os
import sys
from torchvision import transforms
import cmcrameri.cm as cmc
import importlib

from astropy.io import fits


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

    fits_file = ""

    # load arbitrary fits for eda using astropy
    file = "/Users/davidsergio/thesis-org/hst/hst.fits"
    hdul = fits.open(file)
    # print how many hdu's are in the fits file
    print(f"Number of HDUs in the FITS file: {len(hdul)}")
    # print the name of each hdu
    for i, hdu in enumerate(hdul):
        print(f"HDU {i}: {hdu.name}")
        data = hdu.data
        print(f"Data shape: {data.shape if data is not None else 'No data'}")

        # load data into pandas dataframe
        import pandas as pd
        if data is not None:
            df = pd.DataFrame(data)
            print(df.head())

            # print each column name and data type
            print(df.dtypes)

            # plot histogram of ra column if it exists
            if 'ra' in df.columns:
                import matplotlib.pyplot as plt
                plt.hist(df['ra'], bins=50)
                plt.title('Histogram of RA')
                plt.xlabel('RA')
                plt.ylabel('Frequency')
                plt.show()

            if 'dec' in df.columns:
                plt.hist(df['dec'], bins=50)
                plt.title('Histogram of DEC')
                plt.xlabel('DEC')
                plt.ylabel('Frequency')
                plt.show()



if __name__ == "__main__":
    main()
