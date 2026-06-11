from sklearn.base import BaseEstimator, RegressorMixin
from xgboost import XGBRegressor
import numpy as np

class XGBEnsemble(BaseEstimator, RegressorMixin):
    def __init__(self, params_list):
        """
        params_list: list of dicts, each dict is a hyperparameter set for one XGBRegressor
        """
        self.params_list = params_list
        self.models = []

    def fit(self, X, y):
        self.models = []
        for params in self.params_list:
            model = XGBRegressor(**params, random_state=42)
            model.fit(X, y)
            self.models.append(model)
        return self

    def predict(self, X):
        if not self.models:
            raise RuntimeError("You must fit the ensemble before predicting")
        preds = np.column_stack([m.predict(X) for m in self.models])
        return preds.mean(axis=1)  # simple averaging