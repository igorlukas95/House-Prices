from huggingface_hub import dataset_info

from house_price_model.Config.core import _config, TRAINED_MODEL_DIR
import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from pathlib import Path
from house_price_model import __version__
from datasets import load_dataset


def load_datasets(*, mode: str = 'train') -> pd.DataFrame:
    """Loads data as DataFrame

    Args:
        mode (str): Training or testing dataset

    Returns:
        pd.DataFrame: DataFrame with Advanced House Price dataset.
    """

    dataset = load_dataset(_config.config_app.training_data, mode)

    if isinstance(dataset[f'{mode}'], pd.DataFrame):
        return dataset[f'{mode}']

    dataframe = pd.DataFrame([data for data in dataset[f'{mode}']])

    return dataframe


def load_pipeline(*, path: Path) -> Pipeline:
    """Loads saved pipeline

    Args:
        path (Path): Path to pipeline model.

    Returns:
        Pipeline: Pipeline model.
    """
    pipeline = joblib.load(path)
    return pipeline


def save_pipeline(*, pipeline: Pipeline) -> None:
    """Removes a previous version of pipeline and adds new pipeline

    Args:
        pipeline (Pipeline): Pipeline model.

    Returns:
         object:
         None
    """
    model_name = f"{_config.config_app.pipeline_save_file}.{__version__}.pkl"
    model_path = TRAINED_MODEL_DIR / model_name
    remove_pipeline(file_to_keep=[model_name])
    joblib.dump(pipeline, model_path)


def remove_pipeline(*, file_to_keep: t.List[str] = None, model_directory: Path =TRAINED_MODEL_DIR) -> None:
    """Removes old pipeline

    Args:
        model_directory: Directory to TRAINED_MODEL_DIR
        file_to_keep (t.List[str]): Files to keep in directory.

    Returns:
        None
    """

    if file_to_keep is None:
        file_to_keep = []

    for model_path in model_directory.iterdir():
        if model_path.name not in ['__init__.py'] + file_to_keep:
            model_path.unlink()
