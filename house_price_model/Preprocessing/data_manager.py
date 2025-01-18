from house_price_model.Config.core import _config, TRAINED_MODEL_DIR
from house_price_model import __version__
from sklearn.pipeline import Pipeline
from pathlib import Path
import pandas as pd
import typing as t
import joblib
from datasets import load_dataset

def load_datasets(*, mode: str = 'train') -> pd.DataFrame:
    """Loads data as DataFrame

    Args:
        mode (str): Training or testing dataset

    Returns:
        pd.DataFrame: DataFrame with Advanced House Price dataset.
    """

    if not mode in ['train', 'test']:
        raise ValueError("mode must be equal to 'train' or 'test'")

    ds = load_dataset(_config.config_app.dataset, split=mode).to_pandas().set_index('Id')

    if mode == 'test':
        ds.drop('SalePrice', axis=1, inplace=True)

    return ds


def load_pipeline(*, path: Path) -> Pipeline:
    """Loads saved a pipeline

    Args:
        path (Path): Path to a pipeline model.

    Returns:
        Pipeline: Pipeline model.
    """
    pipeline = joblib.load(path)
    return pipeline


def save_pipeline(*, model_pipeline: Pipeline, transformer_pipeline: Pipeline) -> None:
    """Removes a previous version of a pipeline and adds a new pipeline

    Args:
        transformer_pipeline (Pipeline): Transformer Pipeline
        model_pipeline (Pipeline): Model Pipeline

    Returns:
         object:
         None
    """

    model_name = f"{_config.config_app.model_pipeline_save_file}.{__version__}.pkl"
    model_path = TRAINED_MODEL_DIR / model_name

    transformer_name = f"{_config.config_app.preprocessing_pipeline_save_file}.{__version__}.pkl"
    transformer_path = TRAINED_MODEL_DIR / transformer_name

    remove_pipeline(file_to_keep=[model_name, transformer_name])
    joblib.dump(model_pipeline, model_path)
    joblib.dump(transformer_pipeline, transformer_path)


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
