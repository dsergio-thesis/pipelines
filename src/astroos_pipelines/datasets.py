
import numpy as np
import sys
import torch
import os
import importlib
from astropy.io import fits

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from astroos_pipelines.labels import Labels

importlib.reload(sys.modules['astroos_pipelines.labels'])

class DataSetBase(Dataset):
    """
    Base class for datasets.
    """

    def __init__(self, dir):
        super().__init__()
        self.dir = dir
        # create directory if not exists
        os.makedirs(self.dir, exist_ok=True)
        self.filename = os.path.join(self.dir, "dataset.fits")

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
        super().__init__()
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
        images, labels, features, headers = zip(*batch)
        images = torch.stack(images)
        labels = torch.stack(labels)
        features = torch.stack(features)
        headers = list(headers)
        return images, labels, features, headers 


class CutoutDataset(DataSetBase):
    """
    """

    def __init__(self, 
                 m_features_transform=None, 
                 transform=None):
        super().__init__()
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



class FITS_Image_Features_Dataset(DataSetBase):
    """
    FITS Dataset containing band images, photometric features, labels, WCS headers, etc.
    Could possibly include spectra/spectral features in future.
    """

    def __init__(self,
                 dir,
                 labels_init_file=None,
                 N_bands=5,
                 N_features=4,
                 transform=None,
                 photometric_transform=None):
        super().__init__(dir=dir)
        self.transform = transform
        self.hdu_primary = fits.PrimaryHDU()
        self.hdu_list = fits.HDUList([self.hdu_primary])
        self.photometric_transform = photometric_transform
        self.N_bands = N_bands
        self.N_features = N_features

        self.labels = Labels(dir=dir, labels_init_file=labels_init_file)

        self.index = set()
        if self.filename is not None:
            if os.path.exists(self.filename):
                self.hdu_list = fits.open(self.filename)
                for hdu in self.hdu_list[1:]:  # skip primary
                    key = hdu.header['main_id']
                    self.index.add(key)
            else:
                self.hdu_list.writeto(self.filename)
        print("initializing the dataset")

    def __len__(self):
        return len(self.hdu_list) - 1  # Exclude primary HDU

    def __getitem__(self, idx):
        """
        Get item by index. Each item consists of N_bands images, label, photometric features.
        """
        # skip primary HDU
        index = idx + 1

        image = np.array(self.hdu_list[index].data)
        # print(f"calling __getitem__ for index {index}, image shape: {image.shape}")

        # endianness
        if image.dtype.byteorder not in ("=", "|"):
            # image = image.byteswap().newbyteorder()
            image = image.view(image.dtype.newbyteorder('='))

        # contiguous
        image = np.ascontiguousarray(image, dtype=np.float32)
        image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

        image_features = np.zeros((self.N_bands, self.N_features), dtype=np.float32)

        if self.photometric_transform is not None:
            image_features = self.photometric_transform(image)

        label = self.hdu_list[index].header['label']

        if self.transform:
            transformed_image = self.transform(image)
        else:
            transformed_image = image

        # print(f"Returning item index {index}, transformed_image shape: {transformed_image.shape}, label: {label}, image_features shape: {image_features.shape}")

        # return torch.tensor(transformed_image), torch.tensor(label, dtype=torch.long), torch.tensor(image_features), self.hdu_list[index].header
        # return torch.tensor(transformed_image), torch.tensor(label), torch.tensor(image_features), self.hdu_list[index].header
        return torch.tensor(transformed_image, dtype=torch.float32), torch.tensor(label), image_features, self.hdu_list[index].header

    def _contains(self, main_id):
        return main_id in self.index

    def append(self, hdu):
        """
        Append an HDU to the dataset.
        Parameters
        ----------
        hdu : astropy.io.fits.HDU
            HDU to append.
        """

        main_id = hdu.header['main_id']
        print(f"adding {main_id} to the dataset")

        key = main_id

        # Add to index and hdu_list
        self.index.add(key)

        self.hdu_list.append(hdu)
        self.hdu_list.writeto(self.filename, overwrite=True)
    
    def num_classes(self):
        return self.labels.num_classes()
    
    def num_features(self):
        return self.N_features

    def num_bands(self):
        return self.N_bands

    def get_subset(self, indices):
        subset_hdu_list = fits.HDUList([self.hdu_primary])
        for idx in indices:
            index = idx + 1  # skip primary HDU
            subset_hdu_list.append(self.hdu_list[index])
        
        subset_dataset = FITS_Image_Features_Dataset(
            dir=self.dir,
            N_bands=self.N_bands,
            N_features=self.N_features,
            transform=self.transform,
            photometric_transform=self.photometric_transform
        )
        subset_dataset.hdu_list = subset_hdu_list
        subset_dataset.index = {self.hdu_list[idx + 1].header['main_id'] for idx in indices}
        
        return subset_dataset
