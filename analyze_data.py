#!/usr/bin/env python3.7

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


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

    def __init__(self, std_devs_for_outlier=4):
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


if __name__ == '__main__':

    data = pd.read_pickle('pandas/all_data.p')

    pg_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pg']
    pm_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pm']
    pp_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pp']

    #DKPG_data = data.copy(deep=True)
    #for cat in pm_cats + pp_cats:
    #    DKPG_data.pop(cat)

    predictions = ["DK_POINTS", "MIN", "POSS"]
    data_X = data.drop(predictions, axis=1)
    data_Y = data[predictions].copy()

    num_pipeline = Pipeline([
        ('negative_value_remover', NegativeValueRemover()),
        ('outlier_remover', OutlierRemover()),
    ])
    num_attribs = list(data_X)
    num_attribs.remove('HOME')
    cat_attribs = ['HOME']

    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, num_attribs),
        ('cat', OrdinalEncoder(), cat_attribs),
    ])

    data_X_prepared_np = full_pipeline.fit_transform(data_X)
    data_X_prepared = pd.DataFrame(data_X_prepared_np, data_X.index, data_X.columns)

    print(data_X_prepared)
