import numpy as np
import pandas as pd
from typing import Optional, Tuple
from house_price_model.Config.core import _config
from pydantic import BaseModel, ValidationError


def drop_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Drops missing values that are not declared in config"""

    X = X.copy()
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
    """Validates model values"""

    dataframe = X.copy()
    dataframe.rename(columns=_config.config_model.variables_to_rename, inplace=True)
    dataframe = dataframe[_config.config_model.features]
    dataframe = drop_missing_values(dataframe)
    errors = None

    try:
        MainModelConfig(input=dataframe.replace(np.nan, None).to_dict(orient="records"))
    except ValidationError as error:
        errors = error.json()

    return dataframe, errors






class ValidationModelConfig(BaseModel):
    Alley: Optional[str]
    BedroomAbvGr: Optional[int]
    BldgType: Optional[str]
    BsmtCond: Optional[str]
    BsmtExposure: Optional[str]
    BsmtFinSF1: Optional[float]
    BsmtFinSF2: Optional[float]
    BsmtFinType1: Optional[str]
    BsmtFinType2: Optional[str]
    BsmtFullBath: Optional[float]
    BsmtHalfBath: Optional[float]
    BsmtQual: Optional[str]
    BsmtUnfSF: Optional[float]
    CentralAir: Optional[str]
    Condition1: Optional[str]
    Condition2: Optional[str]
    Electrical: Optional[str]
    EnclosedPorch: Optional[int]
    ExterCond: Optional[str]
    ExterQual: Optional[str]
    Exterior1st: Optional[str]
    Exterior2nd: Optional[str]
    Fence: Optional[str]
    FireplaceQu: Optional[str]
    Fireplaces: Optional[int]
    Foundation: Optional[str]
    FullBath: Optional[int]
    Functional: Optional[str]
    GarageArea: Optional[float]
    GarageCars: Optional[float]
    GarageCond: Optional[str]
    GarageFinish: Optional[str]
    GarageQual: Optional[str]
    GarageType: Optional[str]
    GarageYrBlt: Optional[float]
    GrLivArea: Optional[int]
    HalfBath: Optional[int]
    Heating: Optional[str]
    HeatingQC: Optional[str]
    HouseStyle: Optional[str]
    Id: Optional[int]
    KitchenAbvGr: Optional[int]
    KitchenQual: Optional[str]
    LandContour: Optional[str]
    LandSlope: Optional[str]
    LotArea: Optional[int]
    LotConfig: Optional[str]
    LotFrontage: Optional[float]
    LotShape: Optional[str]
    LowQualFinSF: Optional[int]
    MSSubClass: Optional[int]
    MSZoning: Optional[str]
    MasVnrArea: Optional[float]
    MasVnrType: Optional[str]
    MiscFeature: Optional[str]
    MiscVal: Optional[int]
    MoSold: Optional[int]
    Neighborhood: Optional[str]
    OpenPorchSF: Optional[int]
    OverallCond: Optional[int]
    OverallQual: Optional[int]
    PavedDrive: Optional[str]
    PoolArea: Optional[int]
    PoolQC: Optional[str]
    RoofMatl: Optional[str]
    RoofStyle: Optional[str]
    SaleCondition: Optional[str]
    SaleType: Optional[str]
    ScreenPorch: Optional[int]
    Street: Optional[str]
    TotRmsAbvGrd: Optional[int]
    TotalBsmtSF: Optional[float]
    Utilities: Optional[str]
    WoodDeckSF: Optional[int]
    YearBuilt: Optional[int]
    YearRemodAdd: Optional[int]
    YrSold: Optional[int]
    FirstFlrSF: Optional[str]
    SecondFlrSF: Optional[str]
    ThreeSsnPorch: Optional[str]



class MainModelConfig(BaseModel):
    input: ValidationModelConfig
