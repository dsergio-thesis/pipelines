
from abc import abstractmethod
import numpy as np
import sys
import torch
import os
import importlib
import csv
import numpy as np
import torch
from astropy.io import fits
from astropy.wcs import WCS
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from astroos_pipelines.labels import Labels
importlib.reload(sys.modules['astroos_pipelines.labels'])

from astroos_pipelines.utils.formatting import ascii_kv_table
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)

class DataSetBase(Dataset):
    """
    Base class for datasets.

    Parameters
    ----------
    dataset_dir : str
        Directory where the dataset is stored.
    """

    dataset_dir: Path

    def __init__(self, dataset_dir: Path):
        super().__init__()
        self.dataset_dir = dataset_dir
        # create directory if not exists
        os.makedirs(self.dataset_dir, exist_ok=True)
        log.info(f"Dataset directory set to: {self.dataset_dir}")

    def get_dataset_dir(self):
        return self.dataset_dir

    @abstractmethod
    def __len__(self):
        raise NotImplementedError("Must implement __len__ method.")

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError("Must implement __getitem__ method.")

    @abstractmethod
    def append(self, hdu):
        raise NotImplementedError("Must implement append method to add new data to the dataset.")

    @abstractmethod
    def num_classes(self):
        raise NotImplementedError("Must implement num_classes method to return the number of classes in the dataset.")

    @abstractmethod
    def get_labels(self):
        raise NotImplementedError("Must implement get_labels method to return the labels for the dataset.")

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError("Must implement __repr__ method for dataset representation.")

    @abstractmethod
    def contains(self, objectId):
        raise NotImplementedError("Must implement _contains method to check if a main_id is already in the dataset.")

    @abstractmethod
    def update(self, objectId, hdu):
        raise NotImplementedError("Must implement update method to update existing data in the dataset.")


class DataLoaderFITS(DataLoader):
    """
    DataLoader for FITS datasets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to load data from.
    batch_size : int, optional
        How many samples per batch to load (default is 1).
    shuffle : bool, optional
        Set to True to have the data reshuffled at every epoch (default is False).
    num_workers : int, optional
        How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process (default is 1).
    sampler : Sampler, optional
        Defines the strategy to draw samples from the dataset. If specified, shuffle must be False (default is None).
    pin_memory : bool, optional
        If True, the data loader will copy Tensors into CUDA pinned memory before returning them (default is True).
    """

    dateset: Dataset

    def __init__(self, 
                 dataset, 
                 batch_size=1, 
                 shuffle=False, 
                 num_workers=1,
                 sampler=None,
                 pin_memory=True,
                 ):
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=num_workers, 
            sampler=sampler,
            pin_memory=pin_memory,
            collate_fn=self.custom_collate)

    @staticmethod
    def custom_collate(batch: list) -> tuple:
        """ 
        Custom collate function to handle FITS data.
        Expects each item in batch to be a tuple of (image, label, morph_features, phot_features, header).
        """
        images, labels, morph_features, phot_features, headers = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        morph_features = torch.stack(morph_features)
        phot_features = torch.stack(phot_features)
        headers = list(headers)
        return images, labels, morph_features, phot_features, headers 


class FITS_Image_Morphometry_Photometry_Dataset(DataSetBase):
    """
    Dataset class for FITS images with morphometric and photometric features.
    Expects each FITS file to contain:
    - HDU[1] (or named 'CUTOUTS'): multi-band image data
    - HDU[2] (or named 'PHOTO'): photometric features (optional)

    The dataset directory should also contain a manifest.csv file with an 'objectId' column listing the main_id of each object (corresponding to the FITS filenames). The dataset can be initialized with a labels_init_file to set up the labels directory and labels.csv, or you can call dataset.labels.load_from_file() later to load existing labels. The dataset supports appending new objects via the append() method, which saves a new FITS file and updates the manifest.
    
    Parameters
    ----------
    dataset_dir : str
        Directory where the dataset is stored.
    labels_init_file : str, optional
        Path to a CSV file for initializing labels. If provided, this file will be used to create the labels.csv in the dataset directory. If None, labels will not be initialized (default is None).
    N_bands : int, optional
        Number of image bands (default is 5).
    N_morphometric_features : int, optional
        Number of morphometric features to extract (default is 4).
    N_photometric_features : int, optional
        Number of photometric features to extract (default is 4).
    transform : torchvision.transforms.Compose, optional
        Transformations to apply to the image data (default is None).
    morphometric_transform : callable, optional
        Function to extract morphometric features from the image (default is None).
    photometric_transform : callable, optional
        Function to process photometric features (default is None).
    """
    def __init__(self,
                 dataset_dir,
                 labels_init_file=None,
                 N_bands=5,
                 N_morphometric_features=4,
                 N_photometric_features=4,
                 transform=None,
                 morphometric_transform=None,
                 photometric_transform=None):
        super().__init__(dataset_dir=dataset_dir)

        self.transform = transform
        self.photometric_transform = photometric_transform
        self.morphometric_transform = morphometric_transform

        self.N_bands = N_bands
        self.N_morphometric_features = N_morphometric_features
        self.N_photometric_features = N_photometric_features

        if labels_init_file is not None and not os.path.exists(labels_init_file):
            log.error(f"labels_init_file '{labels_init_file}' does not exist")
            raise ValueError(f"labels_init_file '{labels_init_file}' does not exist")
        if labels_init_file is None:
            log.warning("No labels_init_file provided, labels will be empty. Call dataset.labels.load_from_file() later to load labels if needed.")
            self.labels = None
        else:
            self.labels = Labels(labels_dir=dataset_dir, labels_init_file=labels_init_file)

        # Manifest
        self.manifest_file = os.path.join(self.dataset_dir, "manifest.csv")
        self.manifest_list = []
        self.manifest_set = set()

        self._load_manifest()

        log.info(f"initializing dataset {self.dataset_dir} with {len(self.manifest_list)} objects from manifest")

    def _load_manifest(self):
        if not os.path.exists(self.manifest_file):
            return

        with open(self.manifest_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            if "objectId" not in reader.fieldnames:
                raise ValueError(f"manifest.csv must contain an 'objectId' column, got {reader.fieldnames}")

            for row in reader:
                oid = row["objectId"].strip()
                if oid and oid not in self.manifest_set:
                    self.manifest_set.add(oid)
                    self.manifest_list.append(oid)

    def _append_to_manifest_file(self, objectId: str):
        file_exists = os.path.exists(self.manifest_file)

        with open(self.manifest_file, "a", newline="") as f:
            fieldnames = ["objectId"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()

            writer.writerow({"objectId": objectId})

    def __len__(self):
        return len(self.manifest_list)
    
    def num_classes(self):
        if self.labels is None:
            return 0
        return self.labels.num_classes()
    
    def get_labels(self):
        if self.labels is None:
            return None
        return self.labels.get_labels()

    def get_labels_file(self):
        if self.labels is None:
            return None
        return self.labels.get_labels_file()

    def __getitem__(self, idx):
        objectId = self.manifest_list[idx]
        hdul_filename = os.path.join(self.dataset_dir, f"{objectId}.fits")

        with fits.open(hdul_filename, memmap=False) as hdul:

            img_hdu = hdul["CUTOUTS"] if "CUTOUTS" in hdul else None
            pho_hdu = hdul["PHOTO"] if "PHOTO" in hdul else None

            header_dict = dict()

            # Extract header from image HDU if available, otherwise from photometry HDU
            if (img_hdu is not None):
                header_dict = dict(img_hdu.header) if img_hdu.header is not None else {}
            if (img_hdu is None and pho_hdu is not None):
                header_dict = dict(pho_hdu.header) if pho_hdu.header is not None else {}

            # Pixel Data
            if img_hdu is None:
                image = None
            else:
                image = np.array(img_hdu.data, dtype=np.float32)
                # Endianness correction: native byte order 
                if image.dtype.byteorder not in ("=", "|"):
                    image = image.byteswap().newbyteorder()

                wcs = WCS(img_hdu.header)
                print("__getitem__ wcs type:", type(wcs))
                if wcs is not None:
                    print("pixel_n_dim:", getattr(wcs, "pixel_n_dim", None))
                    print("world_n_dim:", getattr(wcs, "world_n_dim", None))
                    if hasattr(wcs, "wcs"):
                        print("CTYPE:", wcs.wcs.ctype)
                header_dict["wcs"] = wcs
            
            # Image Morphometry Transform
            if image is not None and self.morphometric_transform is not None:
                morph = self.morphometric_transform(image).astype(np.float32)
            else:
                morph = None
                # morph = np.zeros((self.N_bands, self.N_morphometric_features), dtype=np.float32)

            # Photometry Data
            if pho_hdu is not None and pho_hdu.data is not None:
                phot = np.array(pho_hdu.data, dtype=np.float32)
            else:
                phot = None
                # phot = np.zeros((self.N_bands, self.N_photometric_features), dtype=np.float32)

            # Photometry Transform
            if phot is not None and self.photometric_transform is not None:
                phot = self.photometric_transform(phot).astype(np.float32)

            # Label Extraction
            if (img_hdu is not None and img_hdu.header is not None) and "label" in img_hdu.header:
                label = int(img_hdu.header.get("label", 0))
            elif (pho_hdu is not None and pho_hdu.header is not None) and "label" in pho_hdu.header:
                label = int(pho_hdu.header.get("label", 0))
            else:
                label = 0

            # Image Pixel Data
            if image is not None and self.transform is not None:
                # if your transform expects torch.Tensor, convert first
                img_for_tf = torch.from_numpy(image)
                img = self.transform(img_for_tf)
            elif image is not None:
                img = torch.from_numpy(image)
            else:
                # img = torch.zeros((self.N_bands, 64, 64), dtype=torch.float32)  # default shape if no image
                img = None

            return (
                img.to(torch.float32) if img is not None else None,
                torch.tensor(label, dtype=torch.long),
                torch.from_numpy(morph).to(torch.float32) if morph is not None else None,
                torch.from_numpy(phot).to(torch.float32) if phot is not None else None,
                header_dict
            )

    def append(self, hdu):
        """
        Append HDU to dataset as a new FITS file. The HDU must contain an 'objectId' in its header which will be used as the main_id and filename for the new entry. The HDU will be saved as a new FITS file in the dataset directory, and the manifest will be updated with the new objectId. If the objectId already exists in the dataset, a ValueError will be raised to prevent duplicate entries.
        """

        objectId  = str(hdu.header["objectId"])
        if self.contains(objectId):
            raise ValueError(f"objectId '{objectId}' already exists in dataset, cannot append duplicate entry.")

        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.append(hdu)
        
        out_path = os.path.join(self.dataset_dir, f"{objectId}.fits")
        print(f"writing item to {out_path}")
        hdul.writeto(out_path, overwrite=True)

        # Update in-memory + on-disk manifest
        self.manifest_set.add(objectId)
        self.manifest_list.append(objectId)
        self._append_to_manifest_file(objectId)

        hdul.close()
    
    def contains(self, objectId):
        """ Check if objectId is already in the dataset """

        objectId = str(objectId)
        # print(f"checking if {objectId} is in self.manifest_set: {self.manifest_set}: {objectId in self.manifest_set}")
        return objectId in self.manifest_set

    def update(self, objectId, hdu):
        """ Update existing FITS file for objectId, adding new HDU """
        if not self.contains(objectId):
            raise ValueError(f"objectId '{objectId}' not found in dataset, cannot update non-existent entry.")

        with fits.open(os.path.join(self.dataset_dir, f"{objectId}.fits"), mode="update") as cur_hdul:
            cur_hdul.append(hdu)
            cur_hdul.flush()

    def __repr__(self):
        info = [
            ("dataset_dir", self.dataset_dir),
            ("num_objects", len(self.manifest_list)),
            ("num_classes", self.num_classes()),
            ("N_bands", self.N_bands),
            ("N_morphometric_features", self.N_morphometric_features),
            ("N_photometric_features", self.N_photometric_features),
            ("manifest", self.manifest_file),
            ("morphometric_transform", ""),
            ("photometric_transform", ""),
        ]
        return ascii_kv_table(info, title="FITS_Image_Morphometry_Photometry_Dataset")


