
import numpy as np
import sys
import torch
import os
import importlib
import csv
import numpy as np
import torch
from astropy.io import fits
from pathlib import Path

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from astroos_pipelines.labels import Labels
importlib.reload(sys.modules['astroos_pipelines.labels'])

from utils.formatting_utils import ascii_kv_table
importlib.reload(sys.modules['utils.formatting_utils'])

from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
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

    def __len__(self):
        raise NotImplementedError("Must implement __len__ method.")

    def __getitem__(self, idx):
        raise NotImplementedError("Must implement __getitem__ method.")


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

    def __getitem__(self, idx):
        objectId = self.manifest_list[idx]
        hdul_filename = os.path.join(self.dataset_dir, f"{objectId}.fits")

        # Always close FITS after reading
        with fits.open(hdul_filename, memmap=False) as hdul:
            # Prefer EXTNAME if you can enforce it; otherwise keep [1]/[2]
            img_hdu = hdul["CUTOUTS"] if "CUTOUTS" in hdul else hdul[1]
            pho_hdu = hdul["PHOTO"]   if "PHOTO"   in hdul else (hdul[2] if len(hdul) > 2 else None)

            image = np.array(img_hdu.data, dtype=np.float32)  # (B,H,W)

            # Endianness safety
            if image.dtype.byteorder not in ("=", "|"):
                image = image.byteswap().newbyteorder()

            # Morphometry
            if self.morphometric_transform is not None:
                morph = self.morphometric_transform(image).astype(np.float32)
            else:
                morph = np.zeros((self.N_bands, self.N_morphometric_features), dtype=np.float32)

            # Photometry
            if pho_hdu is not None and pho_hdu.data is not None:
                phot = np.array(pho_hdu.data, dtype=np.float32)
            else:
                phot = np.zeros((self.N_bands, self.N_photometric_features), dtype=np.float32)

            if self.photometric_transform is not None:
                phot = self.photometric_transform(phot).astype(np.float32)

            label = int(img_hdu.header.get("label", 0))

            # Image transform (torchvision usually expects CHW as torch tensor)
            if self.transform is not None:
                # if your transform expects torch.Tensor, convert first
                img_for_tf = torch.from_numpy(image)
                img = self.transform(img_for_tf)
            else:
                img = torch.from_numpy(image)

            return (
                img.to(torch.float32),
                torch.tensor(label, dtype=torch.long),
                torch.from_numpy(morph).to(torch.float32),
                torch.from_numpy(phot).to(torch.float32),
                dict(img_hdu.header)  # safer for DataLoader collation than Header object
            )

    def append(self, hdul):
        """
        Save one object HDUList to disk and register in manifest.
        Expects:
          hdul[1] = CUTOUTS image HDU (or named 'CUTOUTS')
          hdul[2] = PHOTO image HDU (or named 'PHOTO')
        """
        # Get main_id as string for filenames + CSV
        img_hdu = hdul["CUTOUTS"] if "CUTOUTS" in hdul else hdul[1]
        main_id = str(img_hdu.header["main_id"])

        # Skip if already present
        if main_id in self.manifest_set:
            return

        out_path = os.path.join(self.dataset_dir, f"{main_id}.fits")
        hdul.writeto(out_path, overwrite=True)

        # Update in-memory + on-disk manifest
        self.manifest_set.add(main_id)
        self.manifest_list.append(main_id)
        self._append_to_manifest_file(main_id)
    
    def _contains(self, main_id):
        return main_id in self.manifest_set

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


