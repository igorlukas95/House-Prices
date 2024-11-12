from house_price_model.Config.core import _config, TRAINED_MODEL_DIR, DATASET
import typing as t
import joblib
import pandas as pd
from sklearn.pipeline import  Pipeline
from pathlib import Path
from house_price_model import __version__



def load_dataset(*, filename: Path) -> pd.DataFrame:
    """Loads the data as DataFrame"""
    dataframe = pd.read_csv(f"{DATASET/filename}")
    return dataframe


def load_pipeline(*, filename: Path) -> Pipeline:
    """Loads saved pipeline"""
    path = TRAINED_MODEL_DIR / filename
    pipeline = joblib.load(path)
    return pipeline


def save_pipeline(*, pipeline: Pipeline) -> None:
    """Removes previous pipeline and adds new pipeline"""
    model_name = f"{_config.config_app.pipeline_save_file}.{__version__}.pkl"
    model_path = TRAINED_MODEL_DIR / model_name
    remove_pipeline(file_to_keep=[model_name])
    joblib.dump(pipeline, model_path)


def remove_pipeline(*, file_to_keep: t.List[str] = None) -> None:
    """Removes old pipeline"""
    for model_path  in TRAINED_MODEL_DIR.iterdir():
        if model_path not in ['__init__.py', file_to_keep]:
            model_path.unlink()


