import os.path
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from house_price_model import __version__
from house_price_model.Config.core import _config, TRAINED_MODEL_DIR
from house_price_model.Preprocessing.data_manager import load_datasets, save_pipeline, load_pipeline, remove_pipeline
import pandas as pd
from house_price_model.tests.Mock.dataframe import mock_dataframe


class TestLoadDataset:
    """Test dataset loading functionality"""

    @pytest.fixture
    def test_load_dataset(self, mocker):
        return mocker.patch(
            'house_price_model.Preprocessing.data_manager.load_dataset', autospec=True)

    @pytest.fixture
    def return_mock_dataframe(self, test_load_dataset):
        """Fixture for providing load_dataset a return value as mocked dataframe"""
        test_load_dataset.return_value = {'train': mock_dataframe()}

    @pytest.fixture
    def dataframe(self):
        df = load_datasets()
        return df

    def test_if_loaded_dataframe_type_is_correct(self, test_load_dataset, return_mock_dataframe, dataframe):
        test_load_dataset.assert_called_with(_config.config_app.dataset, 'train')
        assert isinstance(dataframe, pd.DataFrame)

    def test_if_loaded_dataframe_and_mocked_dataframe_are_equal(self, test_load_dataset, return_mock_dataframe,
                                                                dataframe):
        """Test if Dataframe returned by 'load_datasets' matches the mocked dataframe"""
        pd.testing.assert_frame_equal(dataframe, mock_dataframe())

    def test_if_size_of_dataframe_is_correct(self, test_load_dataset, return_mock_dataframe, dataframe):
        assert dataframe.shape == (1460, 81)

    def test_if_loaded_empty_dataframe_is_empty(self, test_load_dataset, dataframe):
        test_load_dataset.return_value = {'train': []}
        test_load_dataset.assert_called_once()
        assert dataframe.empty

    @patch.object(_config.config_app, 'dataset', 'invalid_file')
    def test_if_incorrect_file_path_will_raise_error(self):
        with pytest.raises(FileNotFoundError):
            load_datasets(mode='train')


class TestOperationOnPipelines:

    @pytest.fixture
    def temporary_directory(self):
        with tempfile.TemporaryDirectory() as file:
            yield file

    @patch('joblib.load')
    def test_if_pipeline_is_correctly_loaded(self, mock_pipe):
        pipeline_mock = MagicMock()

        mock_pipe.return_value = pipeline_mock

        mock_file = TRAINED_MODEL_DIR / 'mock_pipe_file.pkl'

        pipeline = load_pipeline(path=Path(mock_file))

        mock_pipe.assert_called_with(Path(TRAINED_MODEL_DIR / mock_file))
        assert pipeline_mock == pipeline

    def test_if_incorrect_file_path_for_load_pipeline_will_raise_error(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline(path=Path('invalid_path'))

    @patch('house_price_model.Preprocessing.data_manager.remove_pipeline')
    @patch('joblib.dump')
    def test_if_pipeline_is_correctly_saved(self, mock_dump, mock_remove):
        pipeline_mock = MagicMock()

        model_name = f'{_config.config_app.pipeline_save_file}.{__version__}.pkl'

        save_pipeline(pipeline=pipeline_mock)

        path = TRAINED_MODEL_DIR / model_name

        mock_dump.assert_called_with(pipeline_mock, path)
        mock_remove.assert_called_once()

    def test_if_pipeline_is_correctly_removed(self, temporary_directory):
        for file in ['file_1.txt', 'file_2.txt', '__init__.py']:
            file_path = Path(temporary_directory) / file
            file_path.touch()

        remove_pipeline(file_to_keep=['file_1.txt'], model_directory=Path(temporary_directory))

        assert os.path.exists(os.path.join(temporary_directory, 'file_1.txt'))
        assert not os.path.exists(os.path.join(temporary_directory, 'file_2.txt'))
        assert os.path.exists(os.path.join(temporary_directory, '__init__.py'))
