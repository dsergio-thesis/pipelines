
from clients.client_local import *

def main():
    
    config, pipeline_metadata = client_config()
    dataset_dir = config.dataset_dir
    dataset_name = config.dataset_name
    label_def_file = config.label_def_file
    pipeline_name = config.pipeline_name
    max_records = config.max_records

    print("Configuration loaded successfully.")

    dataset = FITS_Image_Morphometry_Photometry_Dataset(
            dataset_dir=os.path.join(dataset_dir, dataset_name),
            labels_init_file=label_def_file,
            N_bands=5, 
            N_morphometric_features=4,
            N_photometric_features=4,
            )

    pipelines = [
            PipelineClassification(
                name=pipeline_name,
                metadata=pipeline_metadata,
                max_records=max_records,
                dataset=dataset,
                minor_version=None,
                ),
            ]

            
    pipelines[0].add_stages([ # EDA
        StageHSTCatalogQuery(),
        StageHSTExploratoryDataAnalysis(),
        ])


    pipelines[0].run_pipeline()

    # fits_file = ""

    # # load arbitrary fits for eda using astropy
    # file = "/Users/davidsergio/thesis-org/hst/hst.fits"
    # hdul = fits.open(file)
    # # print how many hdu's are in the fits file
    # print(f"Number of HDUs in the FITS file: {len(hdul)}")
    # # print the name of each hdu
    # for i, hdu in enumerate(hdul):
        # print(f"HDU {i}: {hdu.name}")
        # data = hdu.data
        # print(f"Data shape: {data.shape if data is not None else 'No data'}")

        # # load data into pandas dataframe
        # import pandas as pd
        # if data is not None:
            # df = pd.DataFrame(data)
            # print(df.head())

            # # print each column name and data type
            # print(df.dtypes)

            # # plot histogram of ra column if it exists
            # if 'ra' in df.columns:
                # import matplotlib.pyplot as plt
                # plt.hist(df['ra'], bins=50)
                # plt.title('Histogram of RA')
                # plt.xlabel('RA')
                # plt.ylabel('Frequency')
                # plt.show()

            # if 'dec' in df.columns:
                # plt.hist(df['dec'], bins=50)
                # plt.title('Histogram of DEC')
                # plt.xlabel('DEC')
                # plt.ylabel('Frequency')
                # plt.show()


if __name__ == "__main__":
    main()
