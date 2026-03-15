
from abc import ABC, abstractmethod
import pandas as pd
import os

import sys
import importlib
from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)

class Labels:
    """
    Class to handle loading and managing classification labels from a CSV file.
    """
    
    def __init__(self, 
                 labels_dir, 
                 labels_init_file, 
                 required_columns=None, 
                 unknown_label_index=31):
        
        log.info(f"Initializing Labels with directory: {labels_dir} and init file: {labels_init_file}")
        self.labels_dir = labels_dir
        os.makedirs(self.labels_dir, exist_ok=True)
        self.required_columns = required_columns or ['label_index', 'short_name']
        self.unknown_label_index = unknown_label_index

        if labels_init_file is None:
            self._load_labels()
        else:
            self.labels_init_file = labels_init_file
            if not os.path.exists(self.labels_init_file):
                log.error(f"Labels initialization file not found: {self.labels_init_file}")
                raise FileNotFoundError(f"Labels initialization file not found: {self.labels_init_file}")
            self._initialize_labels()

    def get_labels_file(self):
        return self.labels_init_file

    def _load_labels(self):
        labels_file = os.path.join(self.labels_dir, "labels.csv")
        if os.path.exists(labels_file):
            self.labels = pd.read_csv(labels_file)
            self._validate_labels()
        else:
            log.error(f"Labels file not found: {labels_file}")
            raise FileNotFoundError(f"Labels file not found: {labels_file}")

    def _validate_labels(self):
        """ Validate the labels_file. """

        # validate that the labels has required columns
        if not all(col in self.labels.columns for col in self.required_columns):
            raise ValueError(f"Labels file must contain {self.required_columns} columns. Found columns: {self.labels.columns.tolist()}")

    def _get_label_index(self, short_name):
        if short_name in self.labels.index:
            return self.labels.loc[short_name, 'label_index']
        else:
            return self.unknown_label_index  # Unknown label index

    def _initialize_labels(self):

        log.info(f"Initializing labels from file: {self.labels_init_file}")
        labels = pd.read_csv(self.labels_init_file)

        # add column in the beginning, which is an index
        if ("label_index" not in labels.columns):
            labels.insert(0, 'index', range(0, len(labels)))
        labels.to_csv(self.labels_init_file, index=False)

        # add index for fast lookup
        labels.set_index('short_name', inplace=True, drop=False)
        self.labels = labels
        self._validate_labels()
        self._save_labels()

    def _save_labels(self):
        self.labels.to_csv(f"{self.labels_dir}/labels.csv", index=True)
        log.info(f"Saved labels to {self.labels_dir}/labels.csv")

    def _add_label(self, label):
        if label not in self.labels.index:
            new_index = len(self.labels)
            self.labels.loc[label] = {'index': new_index, 'label_index': new_index}
            self._save_labels()
    
    def get_labels(self):
        return self.labels
    
    def num_classes(self):
        return len(self.labels)

    def to_dict(self):
        d = {}
        d['labels'] = self.labels.to_dict(orient='index')
        d['unknown_label_index'] = self.unknown_label_index
        d['labels_dir'] = self.labels_dir
        return d

    @classmethod
    def from_dict(cls, d):
        instance = cls(labels_dir=d['labels_dir'], labels_init_file=None, unknown_label_index=d['unknown_label_index'])
        instance.labels = pd.DataFrame.from_dict(d['labels'], orient='index')
        return instance


