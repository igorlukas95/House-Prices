import numpy as np
import pandas as pd
from house_price_model.Preprocessing.validation import drop_missing_values, validate_data
from house_price_model.Preprocessing.data_manager import load_pipeline
from house_price_model.Config.core import _config, TRAINED_MODEL_DIR
from house_price_model import __version__
from typing import Union, Dict
from house_price_model.Preprocessing.data_manager import load_datasets
from house_price_model.PreprocessingPipeline import preprocessing_pipeline

pd.set_option('display.max_columns', None)

model_pipe_name = f"{_config.config_app.model_pipeline_save_file}.{__version__}.pkl"
model_pipe = load_pipeline(path=TRAINED_MODEL_DIR / model_pipe_name)

transformer_pipe_name = f"{_config.config_app.preprocessing_pipeline_save_file}.{__version__}.pkl"
transformer_pipe = load_pipeline(path=TRAINED_MODEL_DIR / transformer_pipe_name)

def predict(*, input_data: Union[pd.DataFrame, dict]) -> Dict:
    """Predicts the target variables for given input data

        Args:
            input_data (Union[pd.DataFrame, dict]):
            The input to be predicted, either as DataFrame or a dictionary.

        Returns:
            Dict: Returns dictionary with predictions, errors and version.

    """
    dataframe = pd.DataFrame(input_data)
    validated_data, errors = validate_data(dataframe)
    validated_data = drop_missing_values(validated_data)
    validated_data = validated_data[_config.config_model.features]

    transformed_data = transformer_pipe.transform(validated_data)

    results = {"prediction": [], "errors": errors, "version": __version__}

    y_pred = model_pipe.predict(transformed_data)

    results = {
            "prediction": [np.exp(pred) for pred in y_pred],
            "errors": errors,
            "version": __version__
        }

    return results

if __name__ == "__main__":
    results = predict(input_data=load_datasets(mode='test'))
    print(results['prediction'])




