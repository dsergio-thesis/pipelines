
from astropy.table import Table, vstack
from abc import ABC, abstractmethod
import sys
import os

import importlib

from astroos_pipelines.utils.formatting import ascii_kv_table
importlib.reload(sys.modules['astroos_pipelines.utils.formatting'])

from astroos_pipelines.logger.logger import setup_logging
importlib.reload(sys.modules['astroos_pipelines.logger.logger'])
import logging
setup_logging()
log = logging.getLogger(__name__)


class Pipeline(ABC):
    """
    Abstract base class for data pipelines.
    """

    def __init__(self, name, metadata={}, max_records=None, minor_version=None):
        self.stages = [StageInfo()]
        self.stages[0].stage_name = "Initialization"
        self.stages[0].stage_index = 0

        self._major_version = 2
        self.pipeline_name, self.pipeline_dir = self._construct_name(name, attr={}, minor_version=minor_version)

        self.max_records = max_records
        self.credentials_file = None
    
        os.makedirs(self.pipeline_dir, exist_ok=True)
        self.stages_added = False
        self.metadata = metadata
        self.output = None
        log.debug(f"[PIPELINE] '{self.pipeline_name}' initialized at directory: {self.pipeline_dir}")

    def _construct_name(self, 
                        name: str, 
                        attr: dict[str, str] = None,
                        minor_version: int = None
                        ) -> str:
        """ Construct a unique name for the pipeline based on its attributes. """

        if minor_version is not None:
            pipeline_name = f"{name}_v{self._major_version}.{minor_version}"
        else:
            pipeline_name = f"{name}_v{self._major_version}.0"

        for key, value in attr.items():
            pipeline_name += f"__{key}_{value}"

        pipeline_dir = f"./_pipelines/{pipeline_name}"
        os.makedirs(pipeline_dir, exist_ok=True)
        # find highest numbered subdir v2.x
        existing_dirs = [d for d in os.listdir(f"./_pipelines") if d.startswith(name)]
        if existing_dirs and minor_version is None:
            log.debug([d.split("_v2.") for d in existing_dirs])
            highest_version = max([int(d.split("_v2.")[-1]) for d in existing_dirs])
            pipeline_name = f"{name}_v2.{highest_version + 1}"
            pipeline_dir = f"./_pipelines/{name}_v2.{highest_version + 1}"

        return pipeline_name, pipeline_dir

    @abstractmethod
    def prepare_pipeline(self):
        pass

    def add_stages(self, stages):

        if self.stages_added:
            raise RuntimeError("Stages have already been added to the pipeline.")

        stage_index = self.stages[-1].stage_index + 1 if self.stages else 0
        for stage in stages:

            stage.stage_index = stage_index
            stage_index += 1
            stage.pipeline = self
            stage.pipeline_dir = self.pipeline_dir
            stage.prev_stage = self.stages[-1]
            stage.prev_stage_dir = self.stages[-1].stage_dir

            if stage.requires_stage_dir is True:
                stage.stage_dir = os.path.join(stage.pipeline.pipeline_dir, stage.stage_name)
                os.makedirs(stage.stage_dir, exist_ok=True)
            
            log.debug(f"Adding stage:\n{stage}")
            self.stages.append(stage)

            # stage = StageInfo()
            # stage.stage_index = stage_index
            # stage.prev_stage = self.stages[-1]
            # stage.prev_stage_dir = self.stages[-1].stage_dir
            # stage_index += 1
            # self.stages.append(stage)
        
        self.stages_added = True

    def run_pipeline(self):
        """
        Run prepare_pipeline(), then run() each stage of the pipeline in sequence.
        """
        
        self.prepare_pipeline()

        print(f"---------------- Running Pipeline ---------------- \n")
        log.info(f"running pipeline at directory: {self.pipeline_dir}...")
        for stage in self.stages:
            log.info(f"{stage.stage_index} Running stage: {stage.stage_name}")
            if not stage._validate_prev_stage():
                raise RuntimeError(f"Validation failed for {self.pipeline_name} stage {stage.stage_name}. (_validate_prev_stage returned False)")
            stage.run()
            log.info(f"Completed stage: {stage.stage_name}")
        print("Pipeline completed.")

        self.output = self.stages[-1].output
    
    def clear_pipeline(self):
        """ Clear the pipeline directory if it exists. """
        if os.path.exists(self.pipeline_dir):
            import shutil
            # shutil.rmtree(self.pipeline_dir)
            log.info(f"Cleared pipeline directory: {self.pipeline_dir}")

    def __repr__(self):
        s = f"Pipeline(pipeline_name={self.pipeline_name}, max_records={self.max_records})"
        return s


class PipelineClassification(Pipeline):
    """
    Data pipeline for classification data. 

    Attributes:
        
    """

    def __init__(self,
                 name,
                 max_records,
                 dataset,
                 metadata={},
                 minor_version=None
                 ):
        """ 
        Initialize the data pipeline. 
        
        Parameters:
        """
        super().__init__(name=name, metadata=metadata, max_records=max_records, minor_version=minor_version)
        self.dataset = dataset
        self.X_train_filename = None
        self.y_train_filename = None

    def prepare_pipeline(self):
        pass

    def __repr__(self):
        info = ascii_kv_table([
            ("pipeline_name", self.pipeline_name),
            ("max_records", self.max_records),
            ("dataset", self.dataset.dataset_dir if self.dataset else ""),
            ("X_train_filename", self.X_train_filename),
            ("y_train_filename", self.y_train_filename),
        ], title="PipelineClassification")
        return info


class PipelineDummy(Pipeline):
    """
    Dummy data pipeline for testing.
    """

    def __init__(self, name, max_records=1, metadata={}, minor_version=None):
        super().__init__(name=name, metadata=metadata, max_records=max_records, minor_version=minor_version)
        log.info(f"Initialized dummy pipeline with name: {self.pipeline_name}")

    def _validate_prev_stage(self):
        return True
    
    def run(self):
        log.info("Running Dummy Pipeline...")
        log.info("Dummy Pipeline completed.")

    def prepare_pipeline(self):
        pass


class DataPipelineStage(ABC):
    """
    Abstract base class for data pipeline stages.
    """

    def __init__(self, stage_name, requires_stage_dir=False):
        self.stage_name = stage_name
        self.requires_stage_dir = requires_stage_dir
        self.stage_dir = None
        self.pipeline = None
        self.prev_stage = None
        self.prev_stage_dir = None
        self.output = None

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _validate_prev_stage(self):
        pass

    def __repr__(self):
        s = f"{self.stage_index} DataPipelineStage(stage_name={self.stage_name}, requires_stage_dir={self.requires_stage_dir})\n" \
            f" - requires_stage_dir={self.requires_stage_dir})\n" \
            f" - stage_dir={self.stage_dir}\n" \
            f" - pipeline={self.pipeline}\n" \
            f" - prev_stage={self.prev_stage.stage_name if self.prev_stage else None}\n" \
            f" - prev_stage_dir={self.prev_stage_dir}\n" \
            f" - output={self.output}\n"
        return s

    def cache_pipeline_output(self):
        table = self.output
        # first check cache
        if os.path.exists(f"{self.stage_dir}/output.fits"):
            log.info(f"File {self.stage_dir}/output.fits already exists. ")
            # first read the table
            existing_table = Table.read(f"{self.stage_dir}/output.fits")
            existing_ids = set(existing_table['objectId'])
            mask = [oid not in existing_ids for oid in table['objectId']]
            new_rows = table[mask]
            existing_table = vstack([existing_table, new_rows])
            self.output = existing_table
            existing_table.write(f"{self.stage_dir}/output.fits", format="fits", overwrite=True)
        else:
            table.write(f"{self.stage_dir}/output.fits", format="fits", overwrite=True)
            log.info(f"Saved query result to {self.stage_dir}/output.fits")

        table.write(f"{self.stage_dir}/output.csv", format="csv", overwrite=True)
        log.info(f"Saved query result to {self.stage_dir}/output.csv")

class StageInfo(DataPipelineStage):
    """
    Data pipeline stage for reporting info.
    """

    def __init__(self):
        super().__init__(stage_name="info")

    def _validate_prev_stage(self):
        return True

    def run(self):
        s = f"{self.stage_index} Stage {self.stage_name}"
        if self.prev_stage is not None:
            s += f" reporting on {self.prev_stage.stage_name}: output: {self.prev_stage.output}\n"
        log.info(s)
