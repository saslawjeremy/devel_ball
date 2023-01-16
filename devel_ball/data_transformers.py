from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    MinMaxScaler,
)
from sklearn.compose import ColumnTransformer


class NegativeValueRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove negative values from the features that are now allowed
    to have them.
    """

    def __init__(self, exclude_features=
            ['PLUS_MINUSpg', 'PLUS_MINUSpm', 'PLUS_MINUSpp', 'PER', 'GAME_SCORE', 'PIE',
             'PLUS_MINUStm', 'PLUS_MINUSvsTm']):
        self.exclude_features = exclude_features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for cat in X:
            if cat not in self.exclude_features:
                X.loc[X[X[cat]<0.0][cat].index, cat] = 0
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove outliers from the dataset.
    """

    def __init__(self, std_devs_for_outlier=20):
        self.std_devs_for_outlier = std_devs_for_outlier

    def fit(self, X, y=None):
        self._means = {}
        self._std_devs = {}
        for cat in list(X):
            self._means[cat] = X[cat].mean()
            self._std_devs[cat] = X[cat].std()
        return self

    def transform(self, X):
        for cat in list(X):
            pos_limit = self._means[cat] + self.std_devs_for_outlier * self._std_devs[cat]
            neg_limit = self._means[cat] - self.std_devs_for_outlier * self._std_devs[cat]
            X.loc[X[X[cat] > pos_limit].index, cat] = pos_limit
            X.loc[X[X[cat] < neg_limit].index, cat] = neg_limit
        return X


def get_cleanup_pipeline(data):
    """
    Get the full pipeline to pass training data through to normalize/clean it.

    The current steps are:
        1) Remove negative values from categories that cannot have negative values with NegativeValueRemover
        2) Snip outliers to the boundaries with OutlierRemover
        3) Scale features to a 0 - 1 range with MinMaxScaler
        4) Turn category attributes into numerical attribuets with OrdinalEncoder
    """

    # Remove cat attributes
    num_attribs = list(data)
    num_attribs.remove('HOME')
    cat_attribs = ['HOME']

    num_pipeline = Pipeline([
        ('negative_value_remover', NegativeValueRemover()),
        ('outlier_remover', OutlierRemover()),
        ('min_max_scalar', MinMaxScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OrdinalEncoder(), cat_attribs),
    ])
    return full_pipeline
