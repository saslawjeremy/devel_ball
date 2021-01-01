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
from sklearn.preprocessing import (
    OrdinalEncoder,
    MinMaxScaler,
)


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
    #data = DKPG_data

    data['DK_POINTS_PER_MIN'] = data['DK_POINTS'] / data['MIN']
    data['DK_POINTS_PER_POSS'] = data['DK_POINTS'] / data['POSS']

    predictions = ["DK_POINTS", "MIN", "POSS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS"]
    data_X = data.drop(predictions, axis=1)
    data_Y = data[predictions].copy()

    num_pipeline = Pipeline([
        ('negative_value_remover', NegativeValueRemover()),
        ('outlier_remover', OutlierRemover()),
        ('min_max_scalar', MinMaxScaler()),
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

    from sklearn.linear_model import LinearRegression
    lin_reg = LinearRegression()
    lin_reg.fit(data_X_prepared, data_Y['DK_POINTS'])
    from sklearn.metrics import mean_squared_error
    predictions = lin_reg.predict(data_X_prepared)
    error = mean_squared_error(data_Y['DK_POINTS'], predictions)
    import numpy as np
    print(np.sqrt(error))

    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(lin_reg, data_X_prepared, data_Y['DK_POINTS'], scoring="neg_mean_squared_error", cv=10)
    print(np.mean(np.sqrt(-scores)))
    import IPython; IPython.embed()


    permanent_keys = []
    permanent_value = 1000000000
    while True:
        best_value = 1000000
        best_key = "hi"
        lin_reg = LinearRegression()
        for key in data_X_prepared.keys():
            keys = permanent_keys
            if key not in keys:
                keys.append(key)
            scores = cross_val_score(lin_reg, data_X_prepared[keys], data_Y['DK_POINTS'],
                                     scoring="neg_mean_squared_error", cv=10)
            grade = np.mean(np.sqrt(-scores))
            if grade < best_value:
                best_value = grade
                best_key = key
        if best_value < permanent_value:
            permanent_value = best_value
            permanent_keys.append(best_key)
            print(permanent_value)
            print(permanent_keys)
            lin_reg.fit(data_X_prepared[permanent_keys], data_Y['DK_POINTS'])
            print(lin_reg.coef_)
            print("")
        else:
            break
    
