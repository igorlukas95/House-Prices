import numpy as np
import pandas as pd
from Preprocessing.validation import drop_missing_values, validate_data
from Preprocessing.data_manager import load_pipeline
from Config.core import _config, TRAINED_MODEL_DIR
from house_price_model import __version__
from typing import Union, Dict

pipe_name = f"{_config.config_app.pipeline_save_file}.{__version__}.pkl"

pipeline = load_pipeline(filename=TRAINED_MODEL_DIR / pipe_name)


def predict(*, input_data: Union[pd.DataFrame, dict]) -> Dict:
    """Predict data"""

    dataframe = pd.DataFrame(input_data)

    dataframe, errors = validate_data(dataframe)

    dataframe = drop_missing_values(dataframe)


    y_pred = pipeline.predict(dataframe)

    results = {
        "prediction": [np.exp(pred) for pred in y_pred],
        "errors": errors,
        "version": __version__
    }

    return results











