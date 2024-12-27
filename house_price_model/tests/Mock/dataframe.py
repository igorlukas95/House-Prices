import pandas as pd
import numpy as np



def mock_dataframe(num_rows: int = 1460) -> pd.DataFrame:
    """ Generate a mock DataFrame

    Args:
        num_rows: Number of rows

    Returns:
        DataFrame: Returns dataframe with random generate data similar to original
    """

    np.random.seed(43)

    data = {
        "Id": range(1, num_rows + 1),
        "MSSubClass": np.random.randint(20, 200, size=num_rows),
        "MSZoning": np.random.choice(["RL", "RM", "FV", "RH", "C (all)"], size=num_rows),
        "LotFrontage": np.random.randint(21, 200, size=num_rows),
        "LotArea": np.random.randint(1300, 50000, size=num_rows),
        "Street": np.random.choice(["Pave", "Grvl"], size=num_rows),
        "Alley": np.random.choice(["Pave", "Grvl", np.nan], size=num_rows),
        "LotShape": np.random.choice(["Reg", "IR1", "IR2", "IR3"], size=num_rows),
        "LandContour": np.random.choice(["Lvl", "Bnk", "HLS", "Low"], size=num_rows),
        "Utilities": np.random.choice(["AllPub", "NoSeWa"], size=num_rows),
        "LotConfig": np.random.choice(["Inside", "Corner", "CulDSac", "FR2", "FR3"], size=num_rows),
        "LandSlope": np.random.choice(["Gtl", "Mod", "Sev"], size=num_rows),
        "Neighborhood": np.random.choice(
            ["NAmes", "CollgCr", "OldTown", "Edwards", "Somerst"]),
        "Condition1": np.random.choice(["Norm", "Feedr", "Artery", "RRNn", "PosN", "RRAn", "RRAe"], size=num_rows),
        "Condition2": np.random.choice(["Norm", "Feedr", "Artery", "RRNn", "PosN", "RRAn", "RRAe"], size=num_rows),
        "BldgType": np.random.choice(["1Fam", "TwnhsE", "Duplex", "Twnhs", "2fmCon"], size=num_rows),
        "HouseStyle": np.random.choice(["1Story", "1.5Fin", "1.5Unf", "2Story", "2.5Fin", "2.5Unf", "SFoyer", "SLvl"],
                                   size=num_rows),
        "OverallQual": np.random.randint(1, 11, size=num_rows),
        "OverallCond": np.random.randint(1, 10, size=num_rows),
        "YearBuilt": np.random.randint(1872, 2011, size=num_rows),
        "YearRemodAdd": np.random.randint(1950, 2011, size=num_rows),
        "RoofStyle": np.random.choice(["Gable", "Hip", "Gambrel", "Mansard", "Flat", "Shed"], size=num_rows),
        "RoofMatl": np.random.choice(
            ["CompShg", "WdShngl", "MetalSd", "WdShake", "Membran", "Tar&Grv", "Roll", "ClyTile", "Metal"], size=num_rows),
        "Exterior1st": np.random.choice(
            ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "WdShing", "CemntBd", "Plywood", "AsbShng", "Stucco", "BrkFace",
             "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"], size=num_rows),
        "Exterior2nd": np.random.choice(
            ["VinylSd", "MetalSd", "Wd Sdng", "HdBoard", "Wd Shng", "CemntBd", "Plywood", "AsbShng", "Stucco", "BrkFace",
             "BrkComm", "AsphShn", "Stone", "ImStucc", "CBlock"], size=num_rows),
        "MasVnrType": np.random.choice([np.nan, "BrkFace", "Stone", "BrkCmn", "None"], size=num_rows),
        "MasVnrArea": np.random.randint(0, 1600, size=num_rows),
        "ExterQual": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "ExterCond": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "Foundation": np.random.choice(["PConc", "CBlock", "BrkTil", "Wood", "Slab", "Stone"], size=num_rows),
        "BsmtQual": np.random.choice([np.nan, "Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "BsmtCond": np.random.choice([np.nan, "Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "BsmtExposure": np.random.choice([np.nan, "Gd", "Av", "Mn", "No"], size=num_rows),
        "BsmtFinType1": np.random.choice([np.nan, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf"], size=num_rows),
        "BsmtFinSF1": np.random.randint(0, 1600, size=num_rows),
        "BsmtFinType2": np.random.choice([np.nan, "GLQ", "ALQ", "BLQ", "Rec", "LwQ", "Unf"], size=num_rows),
        "BsmtFinSF2": np.random.randint(0, 1500, size=num_rows),
        "BsmtUnfSF": np.random.randint(0, 1600, size=num_rows),
        "TotalBsmtSF": np.random.randint(0, 6000, size=num_rows),
        "Heating": np.random.choice(["GasA", "GasW", "Grav", "Wall", "Floor"], size=num_rows),
        "HeatingQC": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "CentralAir": np.random.choice(["Y", "N"], size=num_rows),
        "Electrical": np.random.choice(["SBrkr", "FuseA", "FuseF", "FuseP", "Mix"], size=num_rows),
        "1stFlrSF": np.random.randint(300, 4000, size=num_rows),
        "2ndFlrSF": np.random.randint(0, 2000, size=num_rows),
        "LowQualFinSF": np.random.randint(0, 600, size=num_rows),
        "GrLivArea": np.random.randint(300, 5000, size=num_rows),
        "BsmtFullBath": np.random.randint(0, 3, size=num_rows),
        "BsmtHalfBath": np.random.randint(0, 2, size=num_rows),
        "FullBath": np.random.randint(0, 4, size=num_rows),
        "HalfBath": np.random.randint(0, 2, size=num_rows),
        "BedroomAbvGr": np.random.randint(0, 9, size=num_rows),
        "KitchenAbvGr": np.random.randint(0, 3, size=num_rows),
        "KitchenQual": np.random.choice(["Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "TotRmsAbvGrd": np.random.randint(2, 15, size=num_rows),
        "Functional": np.random.choice(["Typ", "Min1", "Min2", "Mod", "Maj", "Sev", "Sal"], size=num_rows),
        "Fireplaces": np.random.randint(0, 3, size=num_rows),
        "FireplaceQu": np.random.choice([np.nan, "Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "GarageType": np.random.choice([np.nan, "Attchd", "Detchd", "BuiltIn", "CarPort", "Basment", "2Types"],
                                       size=num_rows),
        "GarageYrBlt": np.random.randint(1900, 2020, size=num_rows),
        "GarageFinish": np.random.choice([np.nan, "Fin", "RFn", "Unf"], size=num_rows),
        "GarageCars": np.random.randint(0, 4, size=num_rows),
        "GarageArea": np.random.randint(0, 1200, size=num_rows),
        "GarageQual": np.random.choice([np.nan, "Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "GarageCond": np.random.choice([np.nan, "Ex", "Gd", "TA", "Fa", "Po"], size=num_rows),
        "PavedDrive": np.random.choice(["Y", "P", "N"], size=num_rows),
        "WoodDeckSF": np.random.randint(0, 1200, size=num_rows),
        "OpenPorchSF": np.random.randint(0, 600, size=num_rows),
        "EnclosedPorch": np.random.randint(0, 600, size=num_rows),
        "3SsnPorch": np.random.randint(0, 500, size=num_rows),
        "ScreenPorch": np.random.randint(0, 500, size=num_rows),
        "PoolArea": np.random.randint(0, 800, size=num_rows),
        "PoolQC": np.random.choice([np.nan, "Ex", "Gd", "Fa"], size=num_rows),
        "Fence": np.random.choice([np.nan, "GdPrv", "MnPrv", "GdWo", "MnWw"], size=num_rows),
        "MiscFeature": np.random.choice([np.nan, "Shed", "Gar2", "Othr", "TenC"], size=num_rows),
        "MiscVal": np.random.randint(0, 2500, size=num_rows),
        "MoSold": np.random.randint(1, 13, size=num_rows),
        "YrSold": np.random.randint(2006, 2011, size=num_rows),
        "SaleType": np.random.choice(["WD ", "New", "COD", "ConLD", "ConLI", "ConLw", "CWD", "Oth"], size=num_rows),
        "SaleCondition": np.random.choice(["Normal", "Abnorml", "AdjLand", "Alloca", "Family", "Partial"],
                                          size=num_rows),
        "SalePrice": np.random.randint(35000, 750000, size=num_rows)
    }

    return pd.DataFrame(data)