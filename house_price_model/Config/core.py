from pydantic import BaseModel
from typing import List, Mapping, Optional, Dict
from strictyaml import YAML, load
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_FILE_PATH = PACKAGE_ROOT / 'config.yml'
TRAINED_MODEL_DIR = PACKAGE_ROOT / 'trained_models'


class AppConfig(BaseModel):
    """Application-level configuration

    Attributes:
        model (str): Name of model
        dataset (str): Path to dataset
        preprocessing_pipeline_save_file (str): Path name for a pipeline data transformer
        model_pipeline_save_file (str): Path name for a pipeline model
    """
    model: str
    dataset: str
    preprocessing_pipeline_save_file: str
    model_pipeline_save_file: str


class ModelConfig(BaseModel):
    """Model configuration

    Attributes:
        categorical_vars_with_na_frequent (List[str]): List of categorical variables to be imputed using mode.

        categorical_vars_imputing_with_missing (List[str]): List of categorical variables to be imputed using constant 'Missing'

        variables_to_rename (Dict): Dictionary mapping original names to their renamed names.

        temporal_variables (List[str]): List of time-based variables.

        reference_var (str): Time-based variable to be subtracted from temporal_variables.

        log_transformation (List[str]): List of variables for logarithmic transformation.

        binarize_transformation_variables (List[str]): List of variables for binarize transformation.\

        quality_variables (List[str]): List of variables with quality data.

        exposure_variables (List[str]): List of variables with exposure data.

        garagefinish_variables (List[str]): List of variables with quality data related to garage condition.

        categorical_variables_to_encode (List[str]): List of categorical variables to be encoded.

        quality_mapping (Mapping[str]): Dictionary mapping quality variables to numbers.

        exposure_mapping (Mapping[str]): Dictionary mapping exposure variables to numbers.

        garagefinish_mapping (Mapping[str]): Dictionary mapping garage condition to numbers.

        features (List[str]): List of variables.

        alpha (float): Penalization factor.

        test_size (int): Size of testing data.

        random_state (int): Parameter controlling randomness.

        target (str): Parameter to be predicted.

    """
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
    """Combined application and model configuration

    Attributes:
        config_app (AppConfig): Application configuration
        config_model (ModelConfig): Model configuration
    """
    config_app: AppConfig
    config_model: ModelConfig


def load_yaml_file(yaml_path: Optional[Path] = CONFIG_FILE_PATH) -> YAML:
    """Loads a YAML configuration file.

    Args:
        yaml_path (Optional[Path]): Path to YAML file.
        By default, it points to CONFIG_FILE_PATH.

    Returns:
        YAML: YAML content as a Python object.

    Raises:
          OSError: If a file wasn't found.
    """
    try:
        with open(yaml_path, 'r') as file:
            yaml_content = file.read()
        return load(yaml_content)
    except OSError as e:
        print(f"Error opening file {yaml_path}: {e}")


def load_and_validate_config(config_yaml: Path = None) -> Config:
    """Loads YAML object and validates it

    Args:
        config_yaml: yaml path
        By default, its equal to None.

    Returns:
        Config: Validated configuration.
    """

    if config_yaml is None:
        config = load_yaml_file()
    else:
        config = load_yaml_file(yaml_path=config_yaml)

    return Config(
        config_app=AppConfig(**config.data),
        config_model=ModelConfig(**config.data)
    )


_config = load_and_validate_config()
