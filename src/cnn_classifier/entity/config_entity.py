from pathlib import Path
from dataclasses import dataclass
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir:Path
    src_url:str
    local_data_file:Path
    unzip_dir:Path


@dataclass(frozen=True)
class BaseModelPrepConfigEntity:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    # augmentation: bool = True
    params_include_top: bool
    params_epochs: int
    params_classes: int
    params_batch_size: int
    params_img_shape: list 
    params_learning_rate: float 
    params_weights: list




@dataclass(frozen=True)
class ModelTrainingCofigEntity:
    base_model_path:Path
    trained_model_path:Path
    dataset_path:Path
    params_is_augmented:bool
    params_epochs: int
    params_batch_size: int
    params_img_shape: list 
    params_learning_rate: float 
    params_weights: list


