#!/usr/bin/env python3.7

import numpy as np
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
    PolynomialFeatures,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from tensorflow import keras
import talos
import pickle

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


if __name__ == '__main__':

    data = pd.read_pickle('a/all_data.p')

    # Shuffle the data such that it is not in order
    #data = data.sample(frac=1)

    # Remove non-players
    data = data[data['MINpg'] > 10.]
    data = data[data['MIN'] > 0.0]
    data['DK_POINTS_PER_MIN'] = np.where(data['MIN'] == 0.0, 0.0, data['DK_POINTS'] / data['MIN'])
    data['DK_POINTS_PER_POSS'] = np.where(data['POSS'] == 0.0, 0.0, data['DK_POINTS'] / data['POSS'])

    pg_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pg']
    pm_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pm']
    pp_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pp']

    DKPG_data = data.copy(deep=True)
    for cat in pm_cats + pp_cats:
        DKPG_data.pop(cat)
    DKPM_data = data.copy(deep=True)
    for cat in pg_cats + pp_cats:
        DKPM_data.pop(cat)
    DKPP_data = data.copy(deep=True)
    for cat in pg_cats + pm_cats:
        DKPP_data.pop(cat)

    data = DKPG_data
    #data = DKPM_data
    #data = DKPP_data

    #predictions = ["DK_POINTS", "MIN", "POSS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS"]
    predictions = ["DK_POINTS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS"]
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

    data_X_prepared_np = num_pipeline.fit_transform(data_X)
    data_X_prepared = pd.DataFrame(data_X_prepared_np, data_X.index, data_X.columns)
    #data_X_prepared = data_X_prepared[['PTSpg', 'FG3Mpg', 'OREBpg', 'DREBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'MINpg', 'POSSpg']]

    X_train_full, X_test, Y_train_full, Y_test = train_test_split(data_X_prepared, data_Y['DK_POINTS'], train_size=0.85)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, train_size=0.70/0.85)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_full, Y_train_full)
    mae = mean_absolute_error(Y_test, lin_reg.predict(X_test))
    print(mae)

    """
    def build_model(x_train, y_train, x_val, y_val, params):
        input_layer = keras.layers.Input(x_train.shape[1:])
        latest_input_layer = input_layer
        neurons = params['n_neurons']
        for i in range(params['n_hidden']):
            latest_input_layer = keras.layers.Dense(neurons, activation=params['activation_fn'])(latest_input_layer)
            neurons = np.ceil(neurons * params['neuron_decay'])
        if not params['sequential']:
            latest_input_layer = keras.layers.Concatenate()([input_layer, latest_input_layer])
        output = keras.layers.Dense(1)(latest_input_layer)
        model = keras.Model(inputs=[input_layer], outputs=[output])
        optimizer = keras.optimizers.SGD(lr=params['learning_rate'])
        model.compile(loss="mean_absolute_error", optimizer=optimizer)
        early_stopping_cb = keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=True)
        out = model.fit(
            x_train,
            y_train,
            batch_size=params['batch_size'],
            epochs=500,
            validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.TerminateOnNaN(), early_stopping_cb],
        )
        return out, model

    p = {'n_neurons': list(np.arange(10, 200, 10)) + list(np.arange(200, 500, 20)),
         'n_hidden': [1, 2, 3, 4, 5],
         'activation_fn': ['relu', 'tanh'],
         'neuron_decay': [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
         'sequential': [True, False],
         'learning_rate': [0.0001, 0.001, 0.01, 0.1],
         'patience': [10],
         'batch_size': list(np.arange(10, 240, 20)),
    }

    scan = talos.Scan(
        x=X_train,
        y=Y_train,
        model=build_model,
        params=p,
        experiment_name='test',
        x_val=X_valid,
        y_val=Y_valid,
        print_params=True,
        time_limit='2021-1-29 17:00',
        fraction_limit=0.05,
    )

    with open('data.pickle', 'wb') as handle:
        pickle.dump(scan.data, handle)

    import IPython; IPython.embed()
    """

    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(150, activation="relu")(input_)
    hidden2 = keras.layers.Dense(150, activation="relu")(hidden1)
    concat = keras.layers.Concatenate()([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_], outputs=[output])

    sgd = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss="mean_absolute_error", optimizer=sgd)
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
    history = model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=80,
        validation_data=(X_valid, Y_valid),
        callbacks=[early_stopping_cb],
    )
    model.evaluate(X_test, Y_test)

    import IPython; IPython.embed()

    """
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
    """
