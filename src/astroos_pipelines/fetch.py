
from abc import ABC, abstractmethod

import sys
import importlib
from logger.logger import setup_logging
importlib.reload(sys.modules['logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class AstroosFetch(ABC):
    """
    Abstract base class for Astroosfetch clients.
    """

    def __init__(self, label_definitions):
        self.label_definitions = label_definitions

    @abstractmethod
    def fetch_images(self):
        pass

