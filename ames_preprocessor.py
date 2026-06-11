from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

class AmesPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Columns with meaningful categorical nulls
        self.meaningful_nulls_cat = ['PoolQC','MiscFeature','Alley','Fence','FireplaceQu','MasVnrType']
        # Columns with meaningful numerical nulls
        self.meaningful_nulls_num = ['MasVnrArea']
        # Columns with numerical errors
        self.error_nulls = ['LotFrontage']
        # Numerical columns to impute (median or zero)
        self.numerical_null_impute = {
            "LotFrontage": "median",
            "GarageYrBlt": "median",
            "MasVnrArea": "zero",
            "BsmtHalfBath": "zero",
            "BsmtFullBath": "zero",
            "GarageCars": "median",
            "GarageArea": "median",
            "BsmtFinSF2": "zero",
            "BsmtUnfSF": "median",
            "TotalBsmtSF": "median",
            "BsmtFinSF1": "median",
        }
        # Ordinal mappings
        self.df_ordercat_maps = {
            "ExterQual":   {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "ExterCond":   {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "BsmtQual":    {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "BsmtCond":    {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "HeatingQC":   {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "KitchenQual": {"Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "FireplaceQu": {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "GarageQual":  {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
            "GarageCond":  {None:0, np.nan:0, "None":0, "Po":1, "Fa":2, "TA":3, "Gd":4, "Ex":5},
        }
        self.feature_columns_ = None

    def fit(self, X, y=None):
        X_transformed = self.transform(X)
        self.feature_columns_ = X_transformed.columns
        return self


    def transform(self, X):
        X = X.copy()

        # Fill meaningful categorical nulls
        X[self.meaningful_nulls_cat] = X[self.meaningful_nulls_cat].fillna('None')

        # Fill meaningful numerical nulls
        X[self.meaningful_nulls_num] = X[self.meaningful_nulls_num].fillna(0)

        # Fill LotFrontage by neighborhood median
        if 'LotFrontage' in X.columns and 'Neighborhood' in X.columns:
            X['LotFrontage'] = X.groupby('Neighborhood')['LotFrontage'].transform(
                lambda x: x.fillna(x.median())
            )

        # Fill other numerical nulls
        for col, method in self.numerical_null_impute.items():
            if col not in X.columns:
                continue
            if method == "median":
                X[col] = X[col].fillna(X[col].median())
            elif method == "zero":
                X[col] = X[col].fillna(0)

        # Map ordinal categorical columns
        for col, mapping in self.df_ordercat_maps.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(0)

        # One-hot encode remaining categorical columns (drop first to avoid multicollinearity)
        cat_cols = X.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

        # Align columns if fitted
        if self.feature_columns_ is not None:
            X = X.reindex(columns=self.feature_columns_, fill_value=0)

        return X
