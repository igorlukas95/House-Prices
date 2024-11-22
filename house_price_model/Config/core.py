from pydantic import BaseModel
from typing import List, Mapping, Optional, Dict
from strictyaml import YAML, load
from yaml import YAMLError
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
DATASET = PACKAGE_ROOT / 'Dataset'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'


class AppConfig(BaseModel):
    model: str
    training_data: str
    testing_data: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_imputing_with_missing: List[str]
    variables_to_rename: Dict
    temporal_variables: List[str]
    reference_var: str
    log_transformation_variables: List[str]
    binarize_transformation_variables: List[str]
    quality_variables: List[str]
    exposure_variables: List[str]
    garagefinish_variables: List[str]
    categorical_variables_to_encode: List[str]
    quality_mapping: Mapping[str, int]
    exposure_mapping: Mapping[str, int]
    garagefinish_mapping: Mapping[str, int]
    features: List[str]
    alpha: float
    test_size: float
    random_state: int
    target: str


class Config(BaseModel):
    config_app: AppConfig
    config_model: ModelConfig


def load_yaml_file(yaml_path: Optional[Path] = CONFIG_FILE_PATH) -> YAML:
    try:
        with open(yaml_path, 'r') as file:
            yaml_content = file.read()
        return load(yaml_content)
    except OSError as e:
        print(f"Error opening file {yaml_path}: {e}")
    except YAMLError as e:
        print(f"Error parsing YAML file {yaml_path}: {e}")


def load_and_validate_config(config: YAML = None) -> Config:
    config = load_yaml_file()
    return Config(
        config_app=AppConfig(**config.data),
        config_model=ModelConfig(**config.data)
    )


_config = load_and_validate_config()
