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

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Custom transformer to remove outliers from the dataset.
    """

    def __init__(self, std_devs_for_outlier=4):
        self._std_devs_for_outlier = std_devs_for_outlier
        self._means = {}
        self._std_devs = {}

    def fit(self, X, y=None):
        for cat in list(X):
            self._means[cat] = X[cat].mean()
            self._std_devs[cat] = X[cat].std()
        return self

    def transform(self, X):
        import IPython; IPython.embed()
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

    """
    Player traditional stats per game/min/poss + rates:
      all but PLUS_MINUS
        - max (if value > std_dev*4, value = std_dev*4)
        - min (if value < 0, value = 0)
      PLUS_MINUS
        - max and min (std_dev*4)
    """

