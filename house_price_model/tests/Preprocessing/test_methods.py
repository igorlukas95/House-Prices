import numpy as np
import pytest
import pandas as pd
from house_price_model.Preprocessing.methods import (TemporalVariableTransformer, CustomSimpleImpute, Mapper,
                                                     RareLabelsEncoder, MonotonicOrdinalEncoder,
                                                     MathFunctionTransformer,
                                                     CustomBinarizer)
from house_price_model.Config.core import _config


def raises_error(error,
                 transformer,
                 variables,
                 dataframe=None,
                 fit_data=False,
                 y_obligatory=None,
                 target=None,
                 **kwargs):
    if fit_data:
        if y_obligatory:
            with pytest.raises(error):
                transformer(variables=variables, **kwargs).fit_transform(dataframe.drop(target), axis=1)
        else:
            with pytest.raises(error):
                transformer(variables=variables, **kwargs).fit_transform(dataframe)
    else:
        with pytest.raises(error):
            transformer(variables=variables, **kwargs)


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
        raises_error(error=KeyError,
                     transformer=TemporalVariableTransformer,
                     variables=list(sample_datetime_dataframe.columns),
                     reference_str='Invalid_Key',
                     fit_data=True,
                     dataframe=sample_datetime_dataframe)

    def test_if_variable_name_is_invalid(self, sample_datetime_dataframe):
        raises_error(error=KeyError,
                     transformer=TemporalVariableTransformer,
                     variables=['YearBuilt', 'YearRemodAdd', 'InvalidColumn', 'YrSold'],
                     reference_str='YrSold',
                     fit_data=True,
                     dataframe=sample_datetime_dataframe)

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
        return CustomSimpleImpute(variables=list(sample_dataframe.columns),
                                  imputation='mean')

    def test_custom_simple_transformer_impute_invalid_imputation_type(self, transformer, sample_dataframe):
        raises_error(TypeError,
                     CustomSimpleImpute,
                     variables=list(sample_dataframe.columns),
                     imputation=int)

    def test_custom_simple_transformer_invalid_imputation_key(self, sample_dataframe):
        raises_error(ValueError,
                     CustomSimpleImpute,
                     variables=list(sample_dataframe.columns),
                     imputation="invalid")

    def test_custom_simple_transformer_invalid_column_name(self, sample_dataframe, transformer):
        variables_ = ['InvalidQual', 'FireplaceQu']
        raises_error(KeyError, CustomSimpleImpute,
                     variables=variables_,
                     dataframe=sample_dataframe,
                     fit_data=True,
                     imputation="constant")

    @pytest.mark.parametrize("imputation_, fill_values_", get_invalid_parameters())
    def test_custom_simple_transformer_invalid_if_fill_value_type_is_string_and_variable_is_numeric(self,
                                                                                                    sample_dataframe,
                                                                                                    transformer,
                                                                                                    imputation_,
                                                                                                    fill_values_):
        transformer.imputation_ = imputation_
        transformer.fill_values_ = fill_values_

        raises_error(TypeError, transformer=CustomSimpleImpute,
                     variables=list(sample_dataframe.columns),
                     dataframe=sample_dataframe,
                     fit_data=True,
                     imputation=imputation_,
                     fill_values_=fill_values_)

    @pytest.mark.parametrize("imputation_, fill_values_", get_invalid_parameters())
    def test_custom_simple_transformer_raise_error_if_fill_value_is_numeric_and_variable_is_category(self,
                                                                                                     sample_dataframe,
                                                                                                     transformer,
                                                                                                     imputation_,
                                                                                                     fill_values_):
        transformer.imputation_ = imputation_
        transformer.fill_values_ = fill_values_

        raises_error(TypeError,
                     transformer=CustomSimpleImpute,
                     variables=list(sample_dataframe.columns),
                     dataframe=sample_dataframe,
                     fit_data=True,
                     imputation=imputation_,
                     fill_values_=fill_values_)

    def test_custom_simple_transformer_if_calculates_correctly_mean(self, sample_dataframe, transformer):
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
        return Mapper(variables=list(dataframe.columns),
                      mapping=mapping)

    @pytest.mark.parametrize("variables_", ({}, (), "string", 10, 21.4, set()))
    def test_if_mapper_will_raise_error_if_variables_is_not_a_list(self, variables_, mapping):
        raises_error(TypeError,
                     transformer=Mapper,
                     variables=variables_,
                     mapping=mapping)

    @pytest.mark.parametrize("mapping_", ([], (), "mapping", 21, 0.321, set()))
    def test_if_mapper_will_raise_error_if_mapping_passed_value_is_not_a_dict(self, dataframe, mapping_):
        raises_error(TypeError,
                     transformer=Mapper,
                     variables=list(dataframe.columns),
                     mapping=mapping_)

    @pytest.mark.parametrize("mapping_", ({'BsmtQual': '1'}, {2: 'string'}, {3: 31}))
    def test_if_mapper_will_raise_error_if_mapping_keys_are_not_strings_and_mapping_values_an_intiger(self,
                                                                                                      dataframe,
                                                                                                      mapping_):
        raises_error(ValueError,
                     transformer=Mapper,
                     variables=list(dataframe.columns),
                     mapping=mapping_)

    def test_if_mapper_will_raise_error_if_variable_was_not_found_in_dataframe(self, transformer, dataframe, mapping):
        variables_ = ['ExterQual', 'ExterCond', 'BsmtQual', 'Invalid']
        raises_error(KeyError, transformer=Mapper,
                     variables=variables_,
                     dataframe=dataframe,
                     fit_data=True,
                     mapping=mapping)

    def test_if_mapper_maps_correctly(self, transformer, dataframe):
        expected_dataframe = pd.DataFrame(
            {
                'ExterQual': [3, 4, 5, 2, 3, 4, 4],
                'ExterCond': [3, 4, 2, 5, 1, 1, 1],
                'BsmtQual': [3, 4, 5, 2, 0, 3, 1],
                'BsmtCond': [3, 4, 2, 1, 0, 4, 0]
            }
        )

        transformed_dataframe = transformer.fit_transform(dataframe).astype(np.int64)
        pd.testing.assert_frame_equal(transformed_dataframe, expected_dataframe)


class TestRareLabelsEncoder:
    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                'MSZoning': ['RL', 'RL', 'FV', 'FV', 'RH', 'RL', 'RM', 'C (all)', 'FV', 'RH'],
                'LotShape': ['Reg', 'IR1', 'IR2', 'IR1', 'Reg', 'IR1', 'IR2', 'IR3', 'Reg', 'Reg'],
                'LotConfig': ['Inside', 'Inside', 'CulDSac', 'CulDSac', 'FR3', 'Inside', 'Corner', 'CulDSac', 'FR2',
                              'FR2']
            }
        )

    @pytest.mark.parametrize("variables", ({}, set(), 312, 1.2, "invalid"))
    def test_if_encoder_will_raise_error_if_variables_is_not_a_list(self, dataframe, variables):
        raises_error(TypeError,
                     RareLabelsEncoder,
                     variables,
                     threshold=0.01)

    @pytest.mark.parametrize("threshold", ("string", {}, set()))
    def test_if_encoder_will_raise_error_if_threshold_is_not_a_float(self, dataframe, threshold):
        raises_error(TypeError,
                     RareLabelsEncoder,
                     variables=list(dataframe.columns),
                     threshold=threshold)

    @pytest.mark.parametrize("threshold", (12.0, 1.1, -21.0, -0.1))
    def test_if_encoder_will_raise_error_if_threshold_is_out_of_scope(self, dataframe, threshold):
        raises_error(ValueError,
                     RareLabelsEncoder,
                     variables=list(dataframe.columns),
                     threshold=threshold)

    def test_if_encoder_will_raise_error_if_variable_was_not_found_in_dataframe(self, dataframe):
        wrong_variables = ["WoodInvalid", "OpenPorchSF", "BsmtFinSF1"]
        raises_error(KeyError, RareLabelsEncoder,
                     variables=wrong_variables,
                     threshold=0.01,
                     fit_data=True,
                     dataframe=dataframe)

    def test_if_encoder_correctly_encodes_rare_labels(self, dataframe):
        transformer = RareLabelsEncoder(variables=list(dataframe.columns),
                                        threshold=0.2)

        expected_dataframe = pd.DataFrame(
            {
                'MSZoning': ['RL', 'RL', 'FV', 'FV', 'RH', 'RL', 'Rare', 'Rare', 'FV', 'RH'],
                'LotShape': ['Reg', 'IR1', 'IR2', 'IR1', 'Reg', 'IR1', 'IR2', 'Rare', 'Reg', 'Reg'],
                'LotConfig': ['Inside', 'Inside', 'CulDSac', 'CulDSac', 'Rare', 'Inside', 'Rare', 'CulDSac', 'FR2',
                              'FR2']
            }
        )

        transformed_dataframe = transformer.fit_transform(dataframe)

        pd.testing.assert_frame_equal(expected_dataframe, transformed_dataframe)


class TestMonotonicOrdinalEncoder:

    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                'MSZoning': ['RL', 'RL', 'FV', 'FV', 'RH', 'RL', 'RM', 'C (all)', 'FV', 'RH'],
                'LotShape': ['Reg', 'Reg', 'IR2', 'IR2', 'IR1', 'Reg', 'IR2', 'IR3', 'IR2', 'IR1'],
                'LotConfig': ['Inside', 'Inside', 'CulDSac', 'CulDSac', 'FR3', 'Inside', 'Corner', 'FR2', 'CulDSac',
                              'FR2'],
                'SalePrice': [192000, 187000, 160000, 167000, 174000, 199000, 132000, 215000, 165000, 176000]
            }
        )

    @pytest.mark.parametrize("variables_", ({}, set(), "string", 10, ()))
    def test_if_ordinal_encoder_raise_error_if_variables_is_not_a_list(self, variables_):
        raises_error(TypeError,
                     MonotonicOrdinalEncoder,
                     variables=variables_,
                     method="mean")

    @pytest.mark.parametrize("methods_", ({}, set(), 10, 12.3, []))
    def test_if_ordinal_encoder_raise_error_if_method_is_not_a_str(self, methods_, dataframe):
        raises_error(TypeError,
                     MonotonicOrdinalEncoder,
                     variables=list(dataframe.columns),
                     method=methods_)

    def test_if_ordinal_encoder_raise_error_if_method_is_not_median_or_mean(self, dataframe):
        raises_error(ValueError,
                     MonotonicOrdinalEncoder,
                     variables=list(dataframe.columns),
                     method="invalid_method")

    def test_if_ordinal_encoder_raise_error_if_variable_is_not_in_dataframe(self, dataframe):
        variables_ = ["LotShape", "LotConfig", "SalePrice", "MSZoningInvalid"]
        raises_error(KeyError, MonotonicOrdinalEncoder,
                     variables=variables_,
                     method="mean",
                     dataframe=dataframe,
                     fit_data=True,
                     y_obligatory=True,
                     target='SalePrice')

    def test_if_ordinal_encoder_will_correctly_calculate_monotonic_encoder(self, dataframe):
        transformer = MonotonicOrdinalEncoder(variables=list(dataframe.columns),
                                              method="mean")

        expected_dataframe = pd.DataFrame({
            "MSZoning": [3, 3, 1, 1, 2, 3, 0, 4, 1, 2],
            "LotShape": [2, 2, 0, 0, 1, 2, 0, 3, 0, 1],
            "LotConfig": [3, 3, 1, 1, 2, 3, 0, 4, 1, 4],
            "SalePrice": [192000, 187000, 160000, 167000, 174000, 199000, 132000, 215000, 165000, 176000]
        })

        transformed_variables = transformer.fit_transform(dataframe.drop('SalePrice', axis=1), dataframe['SalePrice'])

        pd.testing.assert_frame_equal(expected_dataframe, transformed_variables)


class TestMathFunctionTranformer:
    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                "LotArea": [1000, 2000, 3000, 4000, 5000],
                "FirstFlrSF": [1500, 2500, 3500, 4500, 5500],
                "GrLivArea": [2000, 3000, 4000, 5000, 6000]
            }
        )

    @pytest.fixture
    def transformer(self, dataframe):
        return MathFunctionTransformer(variables=list(dataframe.columns), func="exp")

    @pytest.mark.parametrize("variables_", (set(), {}, 10, 12.1, "string", ()))
    def test_if_math_transformer_will_raise_error_if_variables_are_not_list(self, variables_):
        raises_error(error=TypeError,
                     transformer=MathFunctionTransformer,
                     variables=variables_,
                     func="exp")

    @pytest.mark.parametrize("func_", (set(), {}, 10, 21.1, (), []))
    def test_if_math_transformer_will_raise_error_if_func_is_not_a_string(self, func_, dataframe):
        raises_error(error=TypeError,
                     transformer=MathFunctionTransformer,
                     variables=list(dataframe.columns),
                     func=func_)

    def test_if_math_transformer_raise_error_if_func_is_not_log_or_exp_or_sqrt(self, dataframe):
        raises_error(error=ValueError,
                     transformer=MathFunctionTransformer,
                     variables=list(dataframe.columns),
                     func="invalid")

    def test_if_math_transformer_raise_error_if_variable_is_not_found(self, dataframe):
        raises_error(error=KeyError,
                     transformer=MathFunctionTransformer,
                     variables=['LotArea', 'FirstFlrSF', 'InvalidGrLivArea'],
                     fit_data=True,
                     dataframe=dataframe,
                     func="log")

    def test_if_math_transformer_calculates_logarithm_correctly(self, dataframe, transformer):
        transformer.func_ = "log"
        transformed_data = transformer.fit_transform(dataframe)
        pd.testing.assert_frame_equal(np.log(dataframe), transformed_data)

    def test_if_math_transformer_calculates_exponential_correctly(self, dataframe, transformer):
        transformer.func_ = "exp"
        dataframe = np.sqrt(dataframe)
        transformed_data = transformer.fit_transform(dataframe)
        pd.testing.assert_frame_equal(np.exp(dataframe), transformed_data)

    def test_if_math_transformer_calculates_square_correctly(self, dataframe, transformer):
        transformer.func_ = "sqrt"
        transformed_data = transformer.fit_transform(dataframe)
        pd.testing.assert_frame_equal(np.sqrt(dataframe), transformed_data)


class TestCustomBinarizer:

    @pytest.fixture
    def dataframe(self):
        return pd.DataFrame(
            {
                "BsmtFinSF2": [31, 0, 432, 0, 123, 0, 0, 213],
                "BsmtFinSF1": [312, 321, 0, 0, 231, 0, 21, 32]
            }
        )

    @pytest.mark.parametrize("variables_", (21, 232.1, {}, set(), (), "string"))
    def test_if_custom_binarizer_will_raise_error_if_variables_is_not_a_list(self, dataframe, variables_):
        raises_error(error=TypeError,
                     transformer=CustomBinarizer,
                     variables=variables_,
                     threshold=0.01)

    @pytest.mark.parametrize("threshold_", ({}, [], set(), (), "string"))
    def test_if_custom_binarizer_will_raise_error_if_threshold_is_not_a_number(self, dataframe, threshold_):
        raises_error(error=TypeError,
                     transformer=CustomBinarizer,
                     variables=list(dataframe.columns),
                     threshold=threshold_)

    @pytest.mark.parametrize("threshold_", (-5, -10, -1, 10, 1.2, 1.5, 1000))
    def test_if_custom_binarizer_will_raise_error_if_threshold_is_out_of_range(self, dataframe, threshold_):
        raises_error(error=ValueError,
                     transformer=CustomBinarizer,
                     variables=list(dataframe.columns),
                     threshold=threshold_)


    def test_if_custom_binarizer_will_raise_error_if_variable_is_not_found(self, dataframe):
        raises_error(error=KeyError,
                     transformer=CustomBinarizer,
                     variables=['invalidColumn', 'BsmtFinSF2'],
                     threshold=0.1,
                     fit_data=True,
                     dataframe=dataframe)


    def test_if_custom_binarizer_binarize_correctly(self, dataframe):
        transformer = CustomBinarizer(variables=list(dataframe.columns), threshold=0.1)

