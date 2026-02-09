
from cProfile import label
import numpy as np
import sys
import torch
import os
import importlib
from astropy.io import fits
import os
import csv
import numpy as np
import torch
from astropy.io import fits

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from astroos_pipelines.labels import Labels

importlib.reload(sys.modules['astroos_pipelines.labels'])

class DataSetBase(Dataset):
    """
    Base class for datasets.
    """

    def __init__(self, dataset_dir):
        super().__init__()
        self.dataset_dir = dataset_dir
        # create directory if not exists
        os.makedirs(self.dataset_dir, exist_ok=True)
        # self.filename = os.path.join(self.dir, "dataset.fits")

    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__ method.")

    def __getitem__(self, idx):
        raise NotImplementedError("Subclasses must implement __getitem__ method.")

class SDSSDataset(DataSetBase):
    """
    Custom Dataset for the SDSS Galaxy Survey.

    Parameters
    ----------
    data_tensor : torch.Tensor
        Tensor containing the image data.
    labels_tensor : torch.Tensor
        Tensor containing the labels.
    transform : torchvision.transforms.Compose, optional
        Transformations to apply to each image (default is None).

    Attributes
    ----------
    data_tensor : torch.Tensor
        Stores the image data.
    labels_tensor : torch.Tensor
        Stores the labels.
    transform : torchvision.transforms.Compose or None
        Stores the transformation pipeline.
    """

    def __init__(self, 
                 data_tensor, 
                 labels_tensor, 
                 m_features_transform=None, 
                 transform=None):
        super().__init__(dataset_dir=None)
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor
        self.m_features_transform = m_features_transform
        self.transform = transform

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = self.data_tensor[idx]
        label = self.labels_tensor[idx]
        transformed_sample = sample
        # first apply standard transforms
        if (self.transform is not None):
            transformed_sample = self.transform(sample)

        # then apply morphometric features extraction
        if (self.m_features_transform is not None):
            m_features = self.m_features_transform(transformed_sample)
            # m_features = m_features.view(-1)
        else:
            m_features = torch.tensor([])

        # sample = sample.numpy()
        # m_features = m_features.numpy()
        return transformed_sample, label, m_features

    def get_unique_labels(self):
        return list(set(self.get_labels()))
    
    def summary(self):
        from collections import Counter
        label_counts = Counter(self.get_labels())
        return dict(label_counts)


class DataLoaderFITS(DataLoader):
    """
    DataLoader
    """

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
    def custom_collate(batch):
        images, labels, morph_features, phot_features, headers = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        morph_features = torch.stack(morph_features)
        phot_features = torch.stack(phot_features)
        headers = list(headers)
        return images, labels, morph_features, phot_features, headers 


class CutoutDataset(DataSetBase):
    """
    """

    def __init__(self, 
                 m_features_transform=None, 
                 transform=None):
        super().__init__(dataset_dir=None)
        self.m_features_transform = m_features_transform
        self.transform = transform

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):
        sample = self.data_tensor[idx]
        label = self.labels_tensor[idx]
        transformed_sample = sample
        # first apply standard transforms
        if (self.transform is not None):
            transformed_sample = self.transform(sample)

        # then apply morphometric features extraction
        if (self.m_features_transform is not None):
            m_features = self.m_features_transform(transformed_sample)
            # m_features = m_features.view(-1)
        else:
            m_features = torch.tensor([])

        # sample = sample.numpy()
        # m_features = m_features.numpy()
        return transformed_sample, label, m_features

    def get_unique_labels(self):
        return list(set(self.get_labels()))
    
    def summary(self):
        from collections import Counter
        label_counts = Counter(self.get_labels())
        return dict(label_counts)



# class FITS_Image_Features_Dataset(DataSetBase):
    # """
    # FITS Dataset containing band images, photometric features, labels, WCS headers, etc.
    # Could possibly include spectra/spectral features in future.
    # """

    # def __init__(self,
                 # dir,
                 # labels_init_file=None,
                 # N_bands=5,
                 # N_features=4,
                 # transform=None,
                 # photometric_transform=None):
        # super().__init__(dir=dir)

        # self.transform = transform

        # self.hdu_primary = fits.PrimaryHDU()
        # self.hdu_list = fits.HDUList([self.hdu_primary])

        # self.photometric_transform = photometric_transform
        # self.N_bands = N_bands
        # self.N_features = N_features

        # self.labels = Labels(dir=dir, labels_init_file=labels_init_file)

        # self.index = set()
        # if self.filename is not None:
            # if os.path.exists(self.filename):
                # self.hdu_list = fits.open(self.filename)
                # for hdu in self.hdu_list[1:]:  # skip primary
                    # key = hdu.header['main_id']
                    # self.index.add(key)
            # else:
                # self.hdu_list.writeto(self.filename)
        # print("initializing the dataset")

    # def __len__(self):
        # return len(self.hdu_list) - 1  # Exclude primary HDU

    # def __getitem__(self, idx):
        # """
        # Get item by index. Each item consists of N_bands images, label, photometric features.
        # """
        # # skip primary HDU
        # index = idx + 1

        # image = np.array(self.hdu_list[index].data)
        # print(f"calling __getitem__ for index {index}, image shape: {image.shape}")

        # # image[~np.isfinite(image)] = np.nan
        # # image[image <= -3e38] = np.nan
        # # image[image >=  3e38] = np.nan


        # # endianness
        # if image.dtype.byteorder not in ("=", "|"):
            # # pass
            # # image = image.byteswap().newbyteorder()
            # # image = image.view(image.dtype.newbyteorder('='))
            # image = image.byteswap().newbyteorder()

        # # contiguous
        # # image = np.ascontiguousarray(image, dtype=np.float32)

        # x = image[0,:,:]
        # print("NaNs:", np.isnan(x).sum())
        # print("Infs:", np.isinf(x).sum())
        # print("Finite:", np.isfinite(x).sum())

        # # image = np.nan_to_num(
            # # image,
            # # nan=0.0,
            # # posinf=0.0,
            # # neginf=0.0
        # # )

        # image_b1 = image[0,:,:]
        # print(f"Band 1 - dtype: {image_b1.dtype}, shape: {image_b1.shape}, min: {np.min(image_b1)}, max: {np.max(image_b1)}, mean: {np.mean(image_b1)}, std: {np.std(image_b1)}")
        
        # # print(f"image dtype: {image.dtype}, shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}, mean: {np.mean(image)}, std: {np.std(image)}")
        # # image = np.random.normal(size=image.shape).astype(np.float32)
        # # image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
        # image_features = np.zeros((self.N_bands, self.N_features), dtype=np.float32)

        # if self.photometric_transform is not None:
            # image_features = self.photometric_transform(image)

        # label = self.hdu_list[index].header['label']

        # if self.transform:
            # transformed_image = self.transform(image)
        # else:
            # transformed_image = image

        # # print(f"Returning item index {index}, transformed_image shape: {transformed_image.shape}, label: {label}, image_features shape: {image_features.shape}")

        # # return torch.tensor(transformed_image), torch.tensor(label, dtype=torch.long), torch.tensor(image_features), self.hdu_list[index].header
        # # return torch.tensor(transformed_image), torch.tensor(label), torch.tensor(image_features), self.hdu_list[index].header
        # return torch.tensor(transformed_image, dtype=torch.float32), torch.tensor(label), image_features, self.hdu_list[index].header

    # def _contains(self, main_id):
        # return main_id in self.index

    # def append(self, hdu):
        # """
        # Append an HDU to the dataset.
        # Parameters
        # ----------
        # hdu : astropy.io.fits.HDU
            # HDU to append.
        # """

        # main_id = hdu.header['main_id']
        # # print(f"adding {main_id} to the dataset")

        # key = main_id

        # # Add to index and hdu_list
        # self.index.add(key)

        # self.hdu_list.append(hdu)
        # self.hdu_list.writeto(self.filename, overwrite=True)
    
    # def num_classes(self):
        # return self.labels.num_classes()
    
    # def num_features(self):
        # return self.N_features

    # def num_bands(self):
        # return self.N_bands

    # def get_subset(self, indices):
        # subset_hdu_list = fits.HDUList([self.hdu_primary])
        # for idx in indices:
            # index = idx + 1  # skip primary HDU
            # subset_hdu_list.append(self.hdu_list[index])
        
        # subset_dataset = FITS_Image_Features_Dataset(
            # dir=self.dir,
            # N_bands=self.N_bands,
            # N_features=self.N_features,
            # transform=self.transform,
            # photometric_transform=self.photometric_transform
        # )
        # subset_dataset.hdu_list = subset_hdu_list
        # subset_dataset.index = {self.hdu_list[idx + 1].header['main_id'] for + idx in indices}
        
        # return subset_dataset+ 


class FITS_Image_Morphometry_Photometry_Dataset(DataSetBase):
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
            raise ValueError(f"labels_init_file '{labels_init_file}' does not exist")
        if labels_init_file is None:
            print("Warning: No labels_init_file provided, labels will be empty. Call dataset.labels.load_from_file() later to load labels if needed.")
        else:
            self.labels = Labels(labels_dir=dataset_dir, labels_init_file=labels_init_file)

        # Manifest
        self.manifest_file = os.path.join(self.dataset_dir, "manifest.csv")
        self.manifest_list = []
        self.manifest_set = set()

        self._load_manifest()

        print("initializing the dataset")

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
        return self.labels.num_classes()

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
