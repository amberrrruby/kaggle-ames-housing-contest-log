'''
My ML common function snippets.
These quick utility functions are roughly categorized by usage.
'''
# Load libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import seaborn as sns

def eda_inspect_df(X):
    print(">>> Shape: ", X.shape)
    print(" ===================== ")
    print(">>> Info: ")
    print(X.info())
    print(" ===================== ")
    print(">>> Summary datatypes: ")
    print(X.dtypes.unique())
    print(" ===================== ")
    print(">>> Nulls: ", X.isna().sum())
    print(" ===================== ")
    print(">>> Basic stats:")
    return X.describe()

def eda_catcounts(df, include_na=False):
    """
    Returns a consolidated table of value counts for all categorical columns
    in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    include_na : bool, default False
        Whether to include NaN counts

    Returns
    -------
    pd.DataFrame
        Columns: ['column', 'category', 'count']
    """
    records = []

    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    for col in cat_cols:
        vc = df[col].value_counts(dropna=not include_na)
        for category, count in vc.items():
            records.append({
                "column": col,
                "category": category,
                "count": count
            })

    return pd.DataFrame(records)

def graph_big_hist(X):
    return X.hist(bins=25,figsize=(16,20))

def graph_big_corrmat(X):
    corr = X.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, data=X.corr(), cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()

def graph_three_pairwise_hue(X, hue_col: str=None):
    if not hue_col: sns.pairplot(X, corner=True)
    else: sns.pairplot(X, hue=hue_col, corner=True)
    plt.show()

def graph_onecat_hist(X, col: str, incl_percentage: bool=True):
    values = X[col].value_counts()
    sns.countplot(x=col,data=X,order=values.index)

    if incl_percentage:
        for i in range(values.shape[0]):
            count = values[i] 
            strt='{:0.2f}%'.format(100*count / X.shape[0]) 
            plt.text(i, count+100, strt, ha='center', color='black', fontsize=14) 
    plt.show()

def graph_onenum_histandbox(df, column: str, log1p_transform: bool=False):
    """
    Plots a 2x2 grid of distribution diagnostics for a numeric column:
    - Top-left: Histogram with KDE
    - Bottom-left: CDF
    - Top-right: Box plot
    - Bottom-right: Violin plot

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Name of numeric column to visualize
    log1p_transform : bool
        Whether or not to `log1p` transform the column before plotting
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    data = df[column].dropna()
    if log1p_transform: data = np.log1p(data)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Histogram + KDE (top-left)
    sns.histplot(data, kde=True, ax=axes[0, 0])
    axes[0, 0].set_title("Histogram with KDE")

    # CDF (bottom-left)
    sns.ecdfplot(data, ax=axes[1, 0])
    axes[1, 0].set_title("CDF")

    # Box plot (top-right)
    sns.boxplot(x=data, ax=axes[0, 1])
    axes[0, 1].set_title("Box Plot")

    # Violin plot (bottom-right)
    sns.violinplot(x=data, ax=axes[1, 1])
    axes[1, 1].set_title("Violin Plot")

    plt.tight_layout()
    plt.show()

def graph_two_scatter(df, x_axis: str, y_axis: str, log1p_transform: bool=False):
    if log1p_transform: sns.scatterplot(
        x=np.log1p(df[x_axis]),
        y=df[y_axis],
        alpha=0.2,
        color="green"
    )
    else: sns.scatterplot(x=x_axis, y=y_axis, alpha=0.2, color="green", data=df)

    plt.show()

def graph_three_scatterhue(df, x_axis: str, y_axis: str, hue_col: str):
    sns.scatterplot(x=x_axis, y=y_axis, hue=hue_col, data=df)
    plt.show()

def graph_two_scatter_regr(df, x_axis: str, y_axis: str):
    sns.regplot(x=x_axis, y=y_axis, data=df)
    plt.show()

def graph_two_joint(df, x_axis: str, y_axis: str):
    sns.jointplot(x=x_axis, y=y_axis, data=df)
    plt.show()

def analysis_pcacumvar(X, thres: float=0.9):
    '''***[MYUTILS]*** Plots the cumulative variance graph w/ PCA + 90% (default) cut-off line.'''
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    # Scale numerical features. PCA will cry if you don't
    X_num = X.select_dtypes(include=[np.number])
    # Standardize features (column-wise, vectorized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num)

    # PCA
    pca = PCA()
    pca.fit(X_scaled)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    threshold = thres

    # I'm blind - tell me the numbers
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f'Components needed to reach {int(threshold * 100)}% variance: {n_components}')

    # Plotting
    plt.figure(figsize=(12, 8))
    plt.plot(
        np.arange(1, len(cumulative_variance) + 1),
        cumulative_variance,
        marker='o',
        lw=2
    )

    plt.axhline(y=threshold, linestyle='--')

    plt.xlabel('# Principal Components')
    plt.ylabel('Cumulative explained variance ratio')
    plt.title('PCA cumulative explained variance')
    plt.grid(True)
    plt.show()

def train_and_report_trees(X_train, X_val, y_train, y_val, report_models: bool=True):
    """
    Train common tree-based regression models and report baseline metrics.

    Returns
    -------
    pd.DataFrame
        Columns: ['model', 'mae', 'rmse', 'r2']
    """

    from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

    from sklearn.ensemble import (
        RandomForestRegressor,
        ExtraTreesRegressor,
        GradientBoostingRegressor
    )

    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    # Optional – enable only if installed / needed
    # from catboost import CatBoostRegressor


    tree_models = {
        "random-forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "extra-trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

        "gradient-boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "xgboost": XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_estimators=300,
            learning_rate=0.1,
            verbosity=0
        ),

        "light-gbm": LGBMRegressor(
            random_state=42,
            n_estimators=300,
            learning_rate=0.1
        ),

        # Enable only when categorical features are present
        # and CatBoost is installed
        # "catboost": CatBoostRegressor(
        #     random_state=42,
        #     verbose=False
        # ),
    }


    if report_models:
        print("Training tree models:")
        for name in tree_models:
            print(f" - {name}")

    records = []

    import time
    for name, model in tree_models.items():
        # ---- fit timing ----
        start_fit = time.time()
        model.fit(X_train, y_train)
        fit_time = time.time() - start_fit

        # ---- predict timing ----
        start_pred = time.time()
        preds = model.predict(X_val)
        pred_time = time.time() - start_pred

        # ---- metrics ----
        mae = mean_absolute_error(y_val, preds)
        rmse = root_mean_squared_error(y_val, preds)
        r2 = r2_score(y_val, preds)

        records.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "fit_time_sec": fit_time,
            "pred_time_sec": pred_time
        })

    return pd.DataFrame(records).sort_values("mae")


# === BEGIN ADDED SNIPPETS - REVIEW / STUDY BLOCK ===
def get_grouped_colnames(df: pd.DataFrame) -> dict:
    """Return column names grouped by common dtypes.

    Returns a dict with keys: numeric, categorical, datetime, boolean, other
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime = df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    other = [c for c in df.columns if c not in (numeric + categorical + datetime + boolean)]
    return {
        'numeric': numeric,
        'categorical': categorical,
        'datetime': datetime,
        'boolean': boolean,
        'other': other,
    }

def missing_value_report(df: pd.DataFrame, show_bar: bool = True, show_heatmap: bool = False) -> pd.DataFrame:
    """Return missing counts and percent per column. Optional quick plots.

    - show_bar: show a top-20 barplot of columns with most missing values.
    - show_heatmap: show a small heatmap of missingness (may be slow on large df).
    """
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    out = pd.DataFrame({'missing_count': miss, 'percentage_missing': miss / len(df)})

    if show_bar and not out.empty:
        ax = out.head(20).missing_count.plot(kind='bar', figsize=(10, 4), title='Top missing columns')
        ax.set_ylabel('missing_count')
        plt.tight_layout()
        plt.show()

    if show_heatmap and not out.empty:
        try:
            import seaborn as sns
            subset = df[out.index].isnull().astype(int)
            plt.figure(figsize=(12, 3))
            sns.heatmap(subset.T, cmap='viridis', cbar=False)
            plt.yticks(rotation=0)
            plt.title('Missingness heatmap (columns x rows)')
            plt.show()
        except Exception:
            pass

    return out

def impute_categorical_fill(df: pd.DataFrame, cols, fill_value: str = 'None', tokens=None) -> pd.DataFrame:
    """Robustly fill categorical columns; normalise common NA-like tokens first.

    tokens: optional list of strings to treat as missing (e.g., ['NA','None','nan','']).
    """
    df = df.copy()
    if tokens is None:
        tokens = ['NA', 'None', 'nan', '']

    for c in cols:
        if c not in df.columns:
            continue
        # Normalize a few common textual tokens to np.nan, then fill
        df[c] = df[c].replace(tokens, np.nan)
        df[c] = df[c].fillna(fill_value)
    return df

from sklearn.base import BaseEstimator, TransformerMixin

class GroupMedianImputer(BaseEstimator, TransformerMixin):
    """Impute a single numeric column using the median computed per group.

    Example usage:
        imp = GroupMedianImputer(column='LotFrontage', groupby_col='Neighborhood')
        imp.fit(df)
        df2 = imp.transform(df)
    """
    def __init__(self, column: str, groupby_col: str, fill_value=None):
        self.column = column
        self.groupby_col = groupby_col
        self.fill_value = fill_value

    def fit(self, X, y=None):
        df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        # compute medians per group (dropna so we compute medians on real values)
        self.medians_ = df.groupby(self.groupby_col)[self.column].median().to_dict()
        # global fallback
        self.global_median_ = df[self.column].median() if self.fill_value is None else self.fill_value
        return self

    def transform(self, X):
        df = X.copy()
        if self.column not in df.columns or self.groupby_col not in df.columns:
            return df

        def _impute(row):
            val = row[self.column]
            if pd.notna(val):
                return val
            grp = row[self.groupby_col]
            return self.medians_.get(grp, self.global_median_)

        df[self.column] = df.apply(_impute, axis=1)
        return df

def encode_ordinal(series: pd.Series, mapping: dict, na_value=np.nan) -> pd.Series:
    """Map ordinal strings to integers. Unknowns become NaN (or `na_value`)."""
    return series.map(mapping).fillna(na_value)

# TODO: add other capping methods if needed. read more about this.
def cap_outliers(df: pd.DataFrame, cols, method: str = 'iqr', lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> pd.DataFrame:
    """Cap outliers per column using either 'iqr' or 'quantile' methods."""
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            continue
        ser = df[c]
        if method == 'iqr':
            q1 = ser.quantile(0.25)
            q3 = ser.quantile(0.75)
            iqr = q3 - q1
            lo = q1 - 1.5 * iqr
            hi = q3 + 1.5 * iqr
        else:
            lo = ser.quantile(lower_quantile)
            hi = ser.quantile(upper_quantile)
        df[c] = ser.clip(lower=lo, upper=hi)
    return df

from sklearn.model_selection import StratifiedKFold

def make_cv_splits(y, n_splits: int = 5, bins: int = 10, random_state: int = 42):
    """Create CV splits for regression by binning the target and using StratifiedKFold.

    Returns list of (train_idx, val_idx).
    """
    y = pd.Series(y).reset_index(drop=True)
    # if bins greater than unique values, reduce
    try:
        y_binned = pd.qcut(y, q=min(bins, len(y.unique())), duplicates='drop')
    except Exception:
        y_binned = pd.cut(y, bins=bins)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(X=np.zeros(len(y)), y=y_binned.astype(str)))

def corr_with_target(df: pd.DataFrame, target: str, top_n: int = 20) -> pd.DataFrame:
    """Return top_n numeric correlations with `target`, sorted by absolute value."""
    numeric = df.select_dtypes(include=[np.number])
    if target not in numeric.columns:
        raise ValueError('target not numeric or not in dataframe')
    corrs = numeric.corr()[target].drop(target).abs().sort_values(ascending=False)
    return pd.DataFrame({'feature': corrs.index, 'abs_corr': corrs.values}).head(top_n)

def boxplot_cat_vs_num(df: pd.DataFrame, cat_col: str, num_col: str, figsize=(10, 5)):
    """Show a boxplot of numerical distribution grouped by a categorical column."""
    plt.figure(figsize=figsize)
    sns.boxplot(x=cat_col, y=num_col, data=df)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def qq_and_dist_plot(series: pd.Series, transform: str = None):
    """Plot QQ-plot and distribution for a series. Transform can be 'log1p' or 'boxcox'."""
    ser = series.dropna()
    if transform == 'log1p':
        ser = np.log1p(ser)
    elif transform == 'boxcox':
        # boxcox requires positive values
        from scipy import stats
        if (ser <= 0).any():
            raise ValueError('boxcox requires all positive values')
        ser, _ = stats.boxcox(ser)

    import statsmodels.api as sm
    from scipy.stats import norm
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sm.qqplot(ser, line='45', ax=ax[0])
    sns.histplot(ser, kde=True, stat='density', ax=ax[1])
    x = np.linspace(ser.min(), ser.max(), 100)
    ax[1].plot(x, norm.pdf(x, loc=ser.mean(), scale=ser.std()), color='red')
    ax[0].set_title('QQ-plot')
    ax[1].set_title('Distribution (with normal fit)')
    plt.tight_layout()
    plt.show()

# === END ADDED SNIPPETS - REVIEW / STUDY BLOCK ===

