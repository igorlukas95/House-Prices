import pytest
from house_price_model.Preprocessing.validation import drop_missing_values
import pandas as pd

class TestDropMissingValues:
    @pytest.mark.parametrize("X", ([], (), {}, 32, "string", 32.1))
    def test_if_drop_missing_values_raises_error_if_argument_is_not_a_list(self, X):
        with pytest.raises(TypeError):
                drop_missing_values(X)


    def test_if_drop_missing_values_drops_missing_data(self):
        dataframe = pd.DataFrame({
            "LotArea": [1000, 2000, 3000, 6000, 5000],
            "FirstFlrSF": [1500, 2500, None, 4500, None],
            "GrLivArea": [None, 3000, 4000, 5000, 6000]
        })

        sample_dataframe = dataframe.dropna(subset=["LotArea", "FirstFlrSF", "GrLivArea"])

        test_dataframe = drop_missing_values(X=dataframe)

        pd.testing.assert_frame_equal(sample_dataframe, test_dataframe)



