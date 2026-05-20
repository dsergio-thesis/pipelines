

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

from astroos_pipelines.datasets import DataSetBase
importlib.reload(sys.modules['astroos_pipelines.datasets'])

from astroos_pipelines.labels import Labels
importlib.reload(sys.modules['astroos_pipelines.labels'])

from astroos_pipelines.utils.formatting import ascii_kv_table
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])

# from astroos_pipelines.logger.logger import setup_logging
# importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
# import logging
# setup_logging()
# log = logging.getLogger(__name__)


import torch
from torch.utils.data import DataLoader, Dataset

class DataLoaderComputer(DataLoader):
    """
    DataLoader for Computer_Dataset.
    """

    dataset: Dataset

    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        sampler=None,
        pin_memory=False,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            sampler=sampler,
            pin_memory=pin_memory,
            collate_fn=self.custom_collate,
        )

    @staticmethod
    def custom_collate(batch: list) -> tuple:
        """
        Expects each item in batch to be:
            (e1, e2, e3, e4, e5)

        Returns batched tensors:
            e1_batch, e2_batch, e3_batch, e4_batch, e5_batch
        """
        e1, e2, e3, e4, e5 = zip(*batch)

        def stack_or_empty(items):
            if len(items) == 0:
                return torch.tensor([], dtype=torch.float32)

            first = items[0]

            if isinstance(first, torch.Tensor):
                # Handle placeholder empty tensors like tensor([])
                if first.numel() == 0:
                    return torch.empty((len(items), 0), dtype=first.dtype)
                return torch.stack(items, dim=0)

            return items

        e1 = stack_or_empty(e1)
        e2 = stack_or_empty(e2)
        e3 = stack_or_empty(e3)
        e4 = stack_or_empty(e4)
        e5 = stack_or_empty(e5)

        return e1, e2, e3, e4, e5


import torch

import random
import torch

class Computer_Dataset(DataSetBase):
    """Dataset class for a larger synthetic Buy Computer dataset."""

    def __init__(self, n_samples=200, seed=42):
        super().__init__(dataset_dir="computer_dataset")

        random.seed(seed)

        self.base_feature_names = ["age", "income", "student", "credit"]
        self.target_name = "buys"

        self.categories = {
            "age": ["<=30", "31-40", ">40"],
            "income": ["low", "medium", "high"],
            "student": ["no", "yes"],
            "credit": ["fair", "excellent"],
        }

        self.target_encoder = {"no": 0, "yes": 1}
        self.target_decoder = {0: "no", 1: "yes"}

        self.data = self._generate_data(n_samples)

        self.feature_names = []
        for feature in self.base_feature_names:
            for category in self.categories[feature]:
                self.feature_names.append(f"{feature}_{category}")

        X = []
        y = []
        for row in self.data:
            X.append(self._encode_row(row))
            y.append(self.target_encoder[row[self.target_name]])

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def _generate_data(self, n_samples):
        data = []

        for _ in range(n_samples):
            row = {
                "age": random.choice(self.categories["age"]),
                "income": random.choice(self.categories["income"]),
                "student": random.choice(self.categories["student"]),
                "credit": random.choice(self.categories["credit"]),
            }

            # Simple toy decision rule
            buys = "no"

            if row["age"] == "31-40":
                buys = "yes"
            elif row["age"] == "<=30" and row["student"] == "yes":
                buys = "yes"
            elif row["age"] == ">40" and row["credit"] == "fair":
                buys = "yes"
            else:
                buys = "no"

            # Add a little noise so it's not perfectly deterministic
            if random.random() < 0.1:
                buys = "yes" if buys == "no" else "no"

            row["buys"] = buys
            data.append(row)

        return data

    def _encode_row(self, row):
        encoded = []

        for feature in self.base_feature_names:
            value = row[feature]
            cats = self.categories[feature]

            one_hot = [0.0] * len(cats)
            one_hot[cats.index(value)] = 1.0
            encoded.extend(one_hot)

        return encoded

    def __len__(self):
        return len(self.data)

    def num_classes(self):
        return len(self.target_encoder)

    def get_labels(self):
        return {self.target_name: set(self.target_encoder.keys())}

    def get_labels_file(self):
        return None

    def __getitem__(self, idx):
        return (
            torch.tensor([], dtype=torch.float32),
            self.y[idx],
            torch.tensor([], dtype=torch.float32),
            self.X[idx],
            torch.tensor([], dtype=torch.float32),
        )

    def append(self, item):
        self.data.append(item)
        encoded_x = torch.tensor(self._encode_row(item), dtype=torch.float32)
        encoded_y = torch.tensor(self.target_encoder[item[self.target_name]], dtype=torch.long)

        self.X = torch.cat([self.X, encoded_x.unsqueeze(0)], dim=0)
        self.y = torch.cat([self.y, encoded_y.unsqueeze(0)], dim=0)

    def __repr__(self):
        info = [
            ("Dataset Directory", self.dataset_dir),
            ("Number of Objects", len(self.data)),
            ("Number of Classes", self.num_classes()),
            ("Input Dimension", self.X.shape[1]),
            ("Labels", self.get_labels()),
        ]
        return ascii_kv_table(info, title="Computer_Dataset Summary")

    def to_dict(self):
        return {
            "data": self.data,
            "base_feature_names": self.base_feature_names,
            "target_name": self.target_name,
            "categories": self.categories,
        }

    @classmethod
    def from_dict(cls, d):
        dataset = cls(n_samples=0)
        dataset.data = d["data"]
        dataset.base_feature_names = d["base_feature_names"]
        dataset.target_name = d["target_name"]
        dataset.categories = d["categories"]

        dataset.feature_names = []
        for feature in dataset.base_feature_names:
            for category in dataset.categories[feature]:
                dataset.feature_names.append(f"{feature}_{category}")

        X = []
        y = []
        for row in dataset.data:
            X.append(dataset._encode_row(row))
            y.append(dataset.target_encoder[row[dataset.target_name]])

        dataset.X = torch.tensor(X, dtype=torch.float32)
        dataset.y = torch.tensor(y, dtype=torch.long)

        return dataset
