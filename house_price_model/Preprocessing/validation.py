import numpy as np
import pandas as pd
from typing import Optional, Tuple, List
from pandas.core.interchange.dataframe_protocol import DataFrame
from house_price_model.Config.core import _config
from pydantic import BaseModel, ValidationError, Field


def drop_missing_values(X: pd.DataFrame, unique_missing_columns: bool = False, missing_columns: List[str] = None) -> pd.DataFrame:
    """Drops missing values that aren't declared in config

    Args:
        missing_columns (List[str]): List of missing columns
        unique_missing_columns (bool): If true, user can select their own columns.
        X (pd.DataFrame): Input DataFrame.

    Returns:
        DataFrame: Returns DataFrame with missing data removed.
    """

    if not isinstance(X, pd.DataFrame):
        raise TypeError("Argument must be a DataFrame")

    X = X.copy()

    if unique_missing_columns:
        X.dropna(subset=missing_columns, inplace=True)
    else:
        col_with_missing = [
            col
            for col in _config.config_model.features
            if col not in _config.config_model.categorical_vars_imputing_with_missing
               + _config.config_model.categorical_vars_with_na_frequent
               and X[col].isnull().sum() > 0
        ]
        X.dropna(subset=col_with_missing, inplace=True)

    return X


def validate_data(X: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """Validates model values

    Args:
        X (pd.DataFrame): Input DataFrame

    Returns:
        Tuple[pd.DataFrame, dict]: Returns Tuple with validated dataframe and caught errors.
    """

    dataframe = X.copy()
    dataframe.rename(columns=_config.config_model.variables_to_rename, inplace=True)
    dataframe = dataframe[_config.config_model.features]
    dataframe = drop_missing_values(dataframe)
    errors = {}

    try:
        MainModelConfig(input=dataframe.replace(np.nan, None).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return dataframe, errors

class ValidationModelConfig(BaseModel):
    """ Validation model
    This class defines types and validation rules for each variable.
    """
    Alley: str = Field(max_length=4)
    BedroomAbvGr: int = Field(ge=0, le=7)
    BldgType: str = Field(max_length=6)
    BsmtCond: str = Field(max_length=2)
    BsmtExposure: str = Field(max_length=2)
    BsmtFinSF1: float = Field(ge=0, le=5500)
    BsmtFinSF2: float = Field(ge=0, le=1500)
    BsmtFinType1: str = Field(max_length=3)
    BsmtFinType2: str = Field(max_length=3)
    BsmtFullBath: float = Field(ge=0, le=3)
    BsmtHalfBath: float = Field(ge=0, le=2)
    BsmtQual: str = Field(max_length=2)
    BsmtUnfSF: float = Field(le=0, ge=2350)
    CentralAir: str = Field(max_length=1)
    Condition1: str = Field(max_length=5)
    Condition2: str = Field(max_length=5)
    Electrical: str = Field(max_length=5)
    EnclosedPorch: int = Field(ge=0, le=550)
    ExterCond: str = Field(max_length=2)
    ExterQual: str = Field(max_length=2)
    Exterior1st: str = Field(max_length=7)
    Exterior2nd: str = Field(max_length=7)
    Fence: str = Field(max_length=5)
    FireplaceQu: str = Field(max_length=2)
    Fireplaces: int = Field(ge=0, le=3)
    Foundation: str = Field(max_length=6)
    FullBath: int = Field(ge=0, le=3)
    Functional: str = Field(max_length=5)
    GarageArea: float = Field(ge=0, le=1500)
    GarageCars: float = Field(ge=0, le=5)
    GarageCond: str = Field(max_length=2)
    GarageFinish: str = Field(max_length=3)
    GarageQual: str = Field(max_length=3)
    GarageType: str = Field(max_length=7)
    GarageYrBlt: float = Field(ge=1900, le=2024)
    GrLivArea: int = Field(ge=400, le=5700)
    HalfBath: int = Field(ge=0, le=2)
    Heating: str = Field(max_length=5)
    HeatingQC: str = Field(max_length=2)
    HouseStyle: str = Field(max_length=6)
    Id: int = Field(ge=1)
    KitchenAbvGr: int = Field(ge=0, le=3)
    KitchenQual: str = Field(max_length=2)
    LandContour: str = Field(max_length=3)
    LandSlope: str = Field(max_length=3)
    LotArea: int = Field(ge=1300, le=50000)
    LotConfig: str = Field(max_length=7)
    LotFrontage: float = Field(ge=0, le=150)
    LotShape: str = Field(max_length=3)
    LowQualFinSF: int = Field(ge=0, le=580)
    MSSubClass: int = Field(ge=20, le=190)
    MSZoning: str = Field(max_length=6)
    MasVnrArea: float = Field(ge=0, le=1500)
    MasVnrType: str = Field(max_length=7)
    MiscFeature: str = Field(max_length=4)
    MiscVal: int = Field(ge=0, le=1550)
    MoSold: int = Field(ge=1, le=12)
    Neighborhood: str = Field(max_length=7)
    OpenPorchSF: int = Field(ge=0, le=500)
    OverallCond: int = Field(ge=1, le=9)
    OverallQual: int = Field(ge=1, le=10)
    PavedDrive: str = Field(max_length=1)
    PoolArea: int = Field(le=0, ge=100)
    PoolQC: str = Field(max_length=2)
    RoofMatl: str = Field(max_length=7)
    RoofStyle: str = Field(max_length=7)
    SaleCondition: str = Field(max_length=7)
    SaleType: str = Field(max_length=5)
    ScreenPorch: int = Field(ge=0, le=300)
    Street: str = Field(max_length=4)
    TotRmsAbvGrd: int = Field(ge=2, le=14)
    TotalBsmtSF: float = Field(le=0, ge=2500)
    Utilities: str = Field(max_length=6)
    WoodDeckSF: int = Field(ge=0, le=550)
    YearBuilt: int = Field(ge=1872, le=2024)
    YearRemodAdd: int = Field(ge=1950, le=2024)
    YrSold: int = Field(ge=2010, le=2024)
    FirstFlrSF: int = Field(ge=300, le=3000)
    SecondFlrSF: int = Field(ge=0, le=1500)
    ThreeSsnPorch: int = Field(ge=0, le=250)



class MainModelConfig(BaseModel):
    """This class wraps the ValidationModelConfig to validate a collection of data
    """
    input: ValidationModelConfig




