import numpy as np
import pytest
import pandas as pd
from house_price_model.Preprocessing.methods import TemporalVariableTransformer, CustomSimpleImpute, Mapper
from house_price_model.Config.core import _config


class TestTemporalVariableTransformer:

    @pytest.fixture
    def sample_datetime_dataframe(self):
        return pd.DataFrame(
            {
                'YearBuilt': [1986, 2003, 1999, 1994],
                'YearRemodAdd': [1992, 2004, 2001, 2007],
                'GarageYrBlt': [1993, 2007, 2002, 2008],
                'YrSold': [2008, 2010, 2007, 2010]
            }
        )

    def test_invalid_reference_str_key(self, sample_datetime_dataframe):
        transformer = TemporalVariableTransformer(
            variables=list(sample_datetime_dataframe.drop('YrSold', axis=1).columns), reference_str='invalid_str')
        with pytest.raises(KeyError):
            transformer.fit_transform(sample_datetime_dataframe)

    def test_if_variable_is_not_found(self, sample_datetime_dataframe):
        transformers = TemporalVariableTransformer(variables=['YearBuild', 'GarageYrBlt', 'Invalid_Columns'],
                                                   reference_str='YrSold')
        with pytest.raises(KeyError):
            transformers.fit_transform(sample_datetime_dataframe)

    def test_if_temporal_transformer_calculates_correctly_datetimes(self, sample_datetime_dataframe):
        expected_dataframe = pd.DataFrame(
            {
                'YearBuilt': [22, 7, 8, 16],
                'YearRemodAdd': [16, 6, 6, 3],
                'GarageYrBlt': [15, 3, 5, 2],
                'YrSold': [0, 0, 0, 0]
            }
        )

        transformers = TemporalVariableTransformer(variables=['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold'],
                                                   reference_str='YrSold')
        transformed_dataframe = transformers.fit_transform(sample_datetime_dataframe)
        pd.testing.assert_frame_equal(transformed_dataframe, expected_dataframe)


class TestCustomSimpleImpute:

    @staticmethod
    def get_invalid_parameters():
        return [
            [
                ('constant', 20),
                ('constant', 'Missing'),
            ]
        ]

    @pytest.fixture
    def sample_dataframe(self):
        return pd.DataFrame({
            'BsmtQual': [None, 'Ta', 'Gd', 'Ex', 'Fa', 'Ta', None, 'Ta'],
            'BsmtFinSF1': [421, 0, None, 0, None, 995, 0, 431]
        })

    @pytest.fixture
    def transformer(self, sample_dataframe):
        return CustomSimpleImpute(variables=list(sample_dataframe.columns), imputation='mean')

    def test_custom_simple_transformer_impute_invalid_imputation_type(self, sample_dataframe):
        with pytest.raises(TypeError):
            CustomSimpleImpute(variables=list(sample_dataframe.columns), imputation=int)

    def test_custom_simple_transformer_invalid_imputation_key(self, sample_dataframe):
        with pytest.raises(ValueError):
            CustomSimpleImpute(variables=list(sample_dataframe.columns), imputation='invalid_imputation')

    def test_custom_simple_transformer_invalid_column_name(self, sample_dataframe, transformer):
        sample_dataframe.columns = ['InvalidQual', 'FireplaceQu']
        with pytest.raises(KeyError):
            transformer.fit_transform(sample_dataframe)

    @pytest.mark.parametrize("imputation_, fill_values_", get_invalid_parameters())
    def test_custom_simple_transformer_invalid_if_fill_value_type_is_string_and_variable_is_numeric(self,
                                                                                                    sample_dataframe,
                                                                                                    transformer,
                                                                                                    imputation_,
                                                                                                    fill_values_):
        transformer.imputation_ = imputation_
        transformer.fill_values_ = fill_values_

        with pytest.raises(TypeError):
            CustomSimpleImpute.fit(sample_dataframe['BsmtFinSF1'])

    @pytest.mark.parametrize("imputation_, fill_values_", get_invalid_parameters())
    def test_custom_simple_transformer_invalid_fill_values_if_fill_value_is_numeric_and_variable_is_category(self,
                                                                                                             sample_dataframe,
                                                                                                             transformer,
                                                                                                             imputation_,
                                                                                                             fill_values_):
        transformer.imputation_ = imputation_
        transformer.fill_values_ = fill_values_

        with pytest.raises(TypeError):
            CustomSimpleImpute.fit(sample_dataframe['BsmtQual'])

    def test_custom_simple_tranformer_if_calculates_correctly_mean(self, sample_dataframe, transformer):
        transformer.variables_ = ['BsmtFinSF1']

        transformer.fit(sample_dataframe)

        expected_dataframe = pd.DataFrame({
            'BsmtQual': [None, 'Ta', 'Gd', 'Ex', 'Fa', 'Ta', None, 'Ta'],
            'BsmtFinSF1': [421, 0, 307.8, 0, 307.8, 995, 0, 431]
        })

        assert transformer.encoder_['BsmtFinSF1'] == round(sample_dataframe['BsmtFinSF1'].mean(), 1)

        transformed_dataframe = transformer.transform(sample_dataframe)

        pd.testing.assert_frame_equal(transformed_dataframe, expected_dataframe)

    def test_custom_simple_transformer_if_calculates_correctly_median(self, sample_dataframe, transformer):
        transformer.variables_ = ['BsmtFinSF1']
        transformer.imputation_ = 'median'

        expected_dataframe = pd.DataFrame({
            'BsmtQual': [None, 'Ta', 'Gd', 'Ex', 'Fa', 'Ta', None, 'Ta'],
            'BsmtFinSF1': [421, 0, 210.5, 0, 210.5, 995, 0, 431]
        })

        transformer.fit(sample_dataframe)

        assert transformer.encoder_['BsmtFinSF1'] == sample_dataframe['BsmtFinSF1'].median()

        transformed_dataframe = transformer.transform(sample_dataframe)

        pd.testing.assert_frame_equal(transformed_dataframe, expected_dataframe)

    def test_custom_simple_transformer_if_calculates_correctly_mode(self, sample_dataframe, transformer):
        transformer.variables_ = ['BsmtQual']
        transformer.imputation_ = 'most_frequent'

        expected_dataframe = pd.DataFrame({
            'BsmtQual': ['Ta', 'Ta', 'Gd', 'Ex', 'Fa', 'Ta', 'Ta', 'Ta'],
            'BsmtFinSF1': [421, 0, None, 0, None, 995, 0, 431]
        })

        transformer.fit_transform(sample_dataframe)

        assert transformer.encoder_['BsmtQual'] == sample_dataframe['BsmtQual'].mode()[0]

        transformer_dataframe = transformer.transform(sample_dataframe)

        pd.testing.assert_frame_equal(expected_dataframe, transformer_dataframe)

    def test_custom_simple_transformer_if_calculates_correctly_constant(self, sample_dataframe, transformer):
        transformer.variables_ = ['BsmtQual']
        transformer.imputation_ = 'constant'
        transformer.fill_values_ = 'Missing'

        expected_dataframe = pd.DataFrame({
            'BsmtQual': ['Missing', 'Ta', 'Gd', 'Ex', 'Fa', 'Ta', 'Missing', 'Ta'],
            'BsmtFinSF1': [421, 0, None, 0, None, 995, 0, 431]
        })

        transformer.fit_transform(sample_dataframe)

        assert transformer.encoder_['BsmtQual'] == 'Missing'

        transformer_dataframe = transformer.transform(sample_dataframe)

        pd.testing.assert_frame_equal(expected_dataframe, transformer_dataframe)


class TestMapper:

    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                'ExterQual': ['TA', 'Gd', 'Ex', 'Fa', 'TA', 'Gd', 'Gd'],
                'ExterCond': ['TA', 'Gd', 'Fa', 'Ex', 'Po', 'Po', 'Po'],
                'BsmtQual': ['TA', 'Gd', 'Ex', 'Fa', np.nan, 'TA', 'Po'],
                'BsmtCond': ['TA', 'Gd', 'Fa', 'Po', np.nan, 'Gd', np.nan]
            }
        )

    @pytest.fixture
    def mapping(self):
        return _config.config_model.quality_mapping

    @pytest.fixture
    def transformer(self, dataframe, mapping):
        return Mapper(variables=list(dataframe.columns), mapping=mapping)

    @pytest.mark.parametrize("variables_", ({}, (), "string", 10, 21.4, set()))
    def test_if_function_will_raise_error_if_variables_is_not_a_list(self, variables_, mapping):
        with pytest.raises(TypeError):
            Mapper(variables=variables_, mapping=mapping)

    @pytest.mark.parametrize("mapping_", ([], (), "mapping", 21, 0.321, set()))
    def test_if_mapping_will_raise_error_if_passed_value_is_not_a_dict(self, dataframe, mapping_):
        with pytest.raises(TypeError):
            Mapper(variables=list(dataframe.columns), mapping=mapping_)

    @pytest.mark.parametrize("mapping_", ({'BsmtQual': '1'}, {2: 'string'}, {3: 31}))
    def test_if_function_will_raise_error_if_mapping_keys_are_not_strings_and_mapping_values_an_intigers(self, dataframe,
                                                                                               mapping_):
        with pytest.raises(ValueError):
            Mapper(variables=list(dataframe.columns), mapping=mapping_)

    def test_if_function_will_raise_error_if_variable_was_not_found_in_dataframe(self, transformer, dataframe):
        transformer.variables_ = ['ExterQual', 'ExterCond', 'BsmtQual', 'Invalid']
        with pytest.raises(KeyError):
            transformer.fit_transform(dataframe)



    def test_if_function_maps_correctly(self, transformer, dataframe):

        expected_dataframe = pd.DataFrame(
                {
                    'ExterQual': [3, 4, 5, 2, 3, 4, 4],
                    'ExterCond': [3, 4, 2, 5, 1, 1, 1],
                    'BsmtQual': [3, 4, 5, 2, 0, 3, 1],
                    'BsmtCond': [3, 4, 2, 1, 0, 4, 0]
                }
        )


        transformed_dataframe = transformer.fit_transform(dataframe)
        pd.testing.assert_frame_equal(transformed_dataframe, expected_dataframe)



class TestRareLabelsEncoder:
    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                'BsmtFinSF1': [],
                'WoodDeckSF': [],
                'OpenPorchSF': []
            }
        )
    def test_if_function_will_raise_error_if_variable_are_not_a_list(self):
