# kaggle-ames-housing-contest-log
A log of my attempt for the Kaggle contest "House Prices - Advanced Regression Techniques" from Dec. 2025 (RMSE: 0.12449)

Obtain the dataset [here](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data).

`mlsnips.py` contains some snippets that I have curated to modularize some common code and boilerplate for working with Kaggle datasets. Last updated at Dec 2025.

`ames_preprocessor.py` is a scikit-learn-compatible transformer with `BaseEstimator`, `TransformerMixin` that wraps all the preprocessing for the Ames dataset. Implements `fit`/`transform` so it can eventually fit into a `Pipeline` correctly at inference time.

`xgbtrained.py` is an ensemble wrapper with `BaseEstimator`, `RegressorMixin` that trains multiple `XGBRegressor` instances with different hyperparameter sets and simply averages their predictions.
