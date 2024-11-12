from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
from typing import List, Dict, Union


class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms datetime variables by subtracting a reference datetime variable.

    Attributes:
    variables: List[str]
        List of variables.
    reference_str:  str
        Variable to be subtracted from datetime variables.
    """
    def __init__(self, variables: List[str], reference_str: str) -> None:
        self.variables = variables
        self.reference_str = reference_str

        if not isinstance(variables, list):
            raise TypeError("Variables should be a list")
        if not isinstance(reference_str, str):
            raise TypeError("Reference should be a string")


    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'TemporalVariableTransformer':
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[self.reference_str] - X[col]
            else:
                raise KeyError("Column {col} was not found in DataFrame")
        return X


class CustomSimpleImpute(BaseEstimator, TransformerMixin):
    """
    Impute missing values using specified method.

    Attributes:
        variables: List[str]
            List of variables
        imputation: str
            Imputation method (median, mean, constant, most_frequent)
        fill_values: str
            The fixed value used for filling missing value.
    """
    def __init__(self, variables: List[str], imputation: str = "mean", fill_values: Union[int, float, str] = "Missing"):
        self.variables = variables
        self.imputation = imputation
        self.fill_values = fill_values
        self.encoder: Dict[str, Union[str, float, int]] = {}

        if not isinstance(variables, list):
            raise TypeError("Variables must be a list")
        if self.imputation not in ["mean", "median", "constant", "most_frequent"]:
            raise ValueError("Imputation must be following values ['mean', 'median', 'constant', 'most_frequent']")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomSimpleImpute':
        for col in self.variables:
            if col in X.columns:
                if self.imputation == "mean":
                    self.encoder[col] = X[col].mean()
                elif self.imputation == "median":
                    self.encoder[col] = X[col].median()
                elif self.imputation == "most_frequent":
                    self.encoder[col] = X[col].mode()[0]
                elif self.imputation == "constant":
                    self.encoder[col] = self.fill_values
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].fillna(self.encoder[col])
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return X


class Mapper(BaseEstimator, TransformerMixin):
    """
    Maps categorical variables to numeric values.
    Attributes:
        variables: List[str]
            List of categorical variables
        mapping: Dict[str, int]
            Dictionary with mapping.
    """
    def __init__(self, variables: List[str], mapping: Dict[str, int]) -> None:
        self.variables = variables
        self.mapping = mapping

        if not isinstance(variables, list):
            raise TypeError("Variables is not a list")
        if not isinstance(mapping, dict):
            raise TypeError("Mapping is not a dictionary")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Mapper':
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].map(self.mapping)
            else:
                raise KeyError(f"Column {col} was not found in DataFrame")
        return X



class RareLabelsEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categories with frequency below threshold as Rare.

    Attributes:
        variables: List[str]
            List of variables
        threshold: float
            The minimum frequency required for a variable to be included in returned list
    """
    def __init__(self, variables: List[str], threshold: float = 0.01) -> None:
        self.variables = variables
        self.threshold = threshold
        self.encoder: dict = {}
        if not isinstance(variables, list):
            raise TypeError("Variables is not a list")
        if not isinstance(threshold, float):
            raise ValueError("Threshold must be a floating number")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RareLabelsEncoder':
        X_copy = pd.concat([X, y], axis=1)
        for col in self.variables:
            if col in X.columns:
                rare = (
                    pd.Series(
                        X_copy[col].value_counts(normalize=True)
                    )
                )
                self.encoder[col] = (
                    list(rare[rare > self.threshold].index)
                )
            else:
                raise ValueError(f"Column {col} is missing")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = (
                    np.where(
                        X[col].isin(
                            self.encoder[col]), X[col], 'Rare'
                    )
                )
            else:
                raise ValueError(f"Column {col} is missing")
        return X



class MonotonicOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes categories based on target in ascending order.

    variables: List[str]
        List of variables
    method: str
        Method to calculate order
    """
    def __init__(self, variables: List[str], method: str = 'mean') -> None:
        self.variables = variables
        self.method = method
        self.encoder: Dict[str, dict] = {}
        if not isinstance(variables, list):
            raise TypeError('Variables must be a list')
        if self.method not in ['mean', 'median']:
            raise ValueError('Method must be "median" or "mean"')

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MonotonicOrdinalEncoder':
        dataframe = pd.concat([X, y], axis=1)
        sorted_variables = None
        for col in self.variables:
            if col in X.columns:
                if self.method == 'mean':
                    sorted_variables = (
                        dataframe.groupby([col])['SalePrice']
                        .mean()
                        .sort_values(ascending=True)
                    )
                elif self.method == 'median':
                    sorted_variables = (
                        dataframe.groupby([col])['SalePrice']
                        .median()
                        .sort_values(ascending=True)
                    )

                encoding = {k: i for i, k in enumerate(sorted_variables.index)}
                self.encoder[col] = encoding
        return self


    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = X[col].map(self.encoder[col])
            else:
                raise KeyError(f"Column {col} does not exist")
        return X


class MathFunctionTransformer(BaseEstimator, TransformerMixin):
    """
    Applies a mathematical function to specified variables.

    variables: List[str]
        List of variables
    func: str
        Name of the function to apply ('log', 'sqrt', 'exp', etc.)
    """
    def __init__(self, variables: List[str], func: str):
            self.variables = variables
            self.func = func

            if not isinstance(variables, list):
                raise TypeError("Variables must be a list")
            if not isinstance(func, str):
                raise TypeError("Func must be a str")
            if self.func not in ['log', 'exp', 'sqrt']:
                raise ValueError("Function not supported. Choose from ['log', 'exp', 'sqrt']")

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'MathFunctionTransformer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                if col == 'log':
                    X[col] = np.log(X[col])
                elif col == 'exp':
                    X[col] = np.exp(X[col])
                elif col == 'sqrt':
                    X[col] = np.sqrt(X[col])
            else:
                raise KeyError(f"Column {col} does not exist")
        return X


class CustomBinarizer(BaseEstimator, TransformerMixin):
    """
    Binarizes variables values based on a threshold.

    variables: List[str]
        List of variables
    threshold: float = 0
        Threshold value for binarization
    """
    def __init__(self, variables: List[str], threshold: Union[int, float] = 0):
        self.variables = variables
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: pd.Series = None) -> 'CustomBinarizer':
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.variables:
            if col in X.columns:
                X[col] = np.where(X[col] > 0, 1, 0)
            else:
                raise KeyError(f"Column {col} does not exist")
        return X

