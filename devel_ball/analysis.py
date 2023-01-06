import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from copy import deepcopy
import itertools

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    MinMaxScaler,
    PolynomialFeatures,
)
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from tensorflow import keras
import talos
import pickle
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, LSTM
from keras.callbacks import EarlyStopping

from devel_ball.models import Player
from devel_ball.post_process import get_dk_points


class PredictionType(Enum):
    DKPG = 1 # Draftkings points per game
    DKPM = 2 # Draftkings points per minute
    DKPP = 3 # Draftkings points per possession


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


def cleanup_data(data, data_pipeline=None, prediction_type=PredictionType.DKPG, train=True):
    """
    TODO: docx
    """

    # Remove non-players that we don't want to predict based on. If train, assume we know
    # not to predict players who didn't play in the game
    data = data[data['MINpg'] > 10.]
    if train:
        data = data[data['MIN'] > 0.0]

    ### TODO (JS): Testing adding dk points as a predictive feature here ###
    DK_POINTSpg = get_dk_points(
        data['PTSpg'],
        data['FG3Mpg'],
        data['REBpg'],
        data['ASTpg'],
        data['STLpg'],
        data['BLKpg'],
        data['TOpg'],
    )
    #data['DK_POINTSpg'] = DK_POINTSpg
    ###

    # Create lists of relevant categories
    pg_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pg'] # per-game
    pm_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pm'] # per-minute
    pp_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pp'] # per-possession
    rec_cats = [cat for cat in list(data.columns) if cat[:6] == 'RECENT'] # recent-cats

    # List of categories used for predictions
    predictions = ["DK_POINTS", "MIN", "POSS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS"]

    # Cleanup the recent categories where data may be missing
    unique_recent_cats = set([c[:-1] for c in rec_cats])
    random_recent_cat = list(unique_recent_cats)[0]
    max_lag = max(
        int(c[len(random_recent_cat):]) for c in rec_cats if random_recent_cat in c
    )
    for rec_cat in unique_recent_cats:
        for i in range(1, max_lag + 1):
            data['{}{}'.format(rec_cat, i)] = (
                np.where(
                    data['{}{}'.format(rec_cat, i)].isnull(),
                    data['{}{}'.format(rec_cat, i-1)],
                    data['{}{}'.format(rec_cat, i)]
                )
            )
        # Try adding different recent averages (last 3 games, 5 games, 10 games, etc.)
        #data[f'{rec_cat}_last_3_average'] = (data[f'{rec_cat}0'] + data[f'{rec_cat}1'] + data[f'{rec_cat}2']) / 3.
    # TODO (JS): try cleaning up recent stats with averages, max/min, etc.


    # Create dataframes pertaining to which type of data to predict from
    DKPG_data = data.copy(deep=True) # draftkings points per game prediction
    for cat in pm_cats + pp_cats:
        DKPG_data.pop(cat)
    DKPM_data = data.copy(deep=True) # draftkings points per minute prediction
    # Need to create recent per_min and per_poss stats to use below
    for cat in pg_cats + pp_cats + rec_cats:
        DKPM_data.pop(cat)
    DKPP_data = data.copy(deep=True) # draftkings points per possession prediction
    for cat in pg_cats + pm_cats + rec_cats:
        DKPP_data.pop(cat)

    # Select the right dataframe based on prediction type
    data = {
        PredictionType.DKPG: DKPG_data,
        PredictionType.DKPM: DKPM_data,
        PredictionType.DKPP: DKPP_data,
    }[prediction_type]

    # Separate prediction data from results data
    data_X = data.drop(predictions, axis=1)
    data_Y = data[predictions].copy()
    predict = {
        PredictionType.DKPG: 'DK_POINTS',
        PredictionType.DKPM: 'DK_POINTS_PER_MIN',
        PredictionType.DKPP: 'DK_POINTS_PER_POSS',
    }[prediction_type]
    data_Y = data_Y[predict]

    # Store accounting data
    accounting_cats = ['PLAYER_ID', 'DATE']
    data_accounting = data_X[accounting_cats].copy()
    data_X = data_X.drop(accounting_cats, axis=1)

    # Get the pipeline and clean the data
    if data_pipeline is None:
        data_pipeline = get_cleanup_pipeline(data_X)
        data_X_prepared_np = data_pipeline.fit_transform(data_X)
    else:
        data_X_prepared_np = data_pipeline.transform(data_X)
    output_columns = list(
        itertools.chain.from_iterable([t[2] for t in data_pipeline.transformers_])
    )
    data_X_prepared = pd.DataFrame(data_X_prepared_np, data_X.index, output_columns)
    return data_X_prepared, data_Y, data_accounting, data_pipeline


    ### TEMP - LEFTOVER NOTES FROM TESTING RECENCY DATA ON MINS IT SEEMS ###
    # Create 3D tensor
    #tensor = []
    #for i, row in REC_data.iterrows():
    #    d = np.zeros((10, 1))
    #    for index in range(10):
    #        d[index][0] = row['RECENT_MIN{}'.format(index)]
    #    tensor.append(d)
    #X = np.asarray(tensor)
    #Y = REC_data['MIN']
    ## Create model
    #model = Sequential()
    #model.add(LSTM(128, input_shape=((10, 1))))
    #model.add(Dense(1))
    #model.compile(optimizer='RMSprop', loss='mean_absolute_error')
    #earlystop = EarlyStopping(monitor="loss", min_delta=0, patience=5)
    #model.fit(X, Y, batch_size=32, epochs=10, callbacks=[earlystop])
    #####################################################################


def get_model(data):

    # Get data
    prediction_type = PredictionType.DKPG
    data_X, data_Y, data_accounting, data_pipeline = cleanup_data(
        data, prediction_type=prediction_type, train=True
    )
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        data_X, data_Y, train_size=0.85,
        shuffle=False, random_state=42,  # Uncomment for deterministic testing
    )
    X_train, X_valid, Y_train, Y_valid = train_test_split(
        X_train_full, Y_train_full, train_size=0.70/0.85,
        shuffle=False, random_state=42,  # Uncomment for deterministic testing
    )

    # Create model
    model = get_regression_model(X_train_full, Y_train_full)
    #model = get_neural_net_model(X_train, Y_train, X_valid, Y_valid)
    print_model(model)

    # Get baseline by just looking at average dk points
    raw_test_data = data[data.index.isin(X_test.index)]
    letter = {
        PredictionType.DKPG: 'g', PredictionType.DKPM: 'm', PredictionType.DKPP: 'p'
    }[prediction_type]
    baseline = get_dk_points(
        raw_test_data[f'PTSp{letter}'],
        raw_test_data[f'FG3Mp{letter}'],
        raw_test_data[f'REBp{letter}'],
        raw_test_data[f'ASTp{letter}'],
        raw_test_data[f'STLp{letter}'],
        raw_test_data[f'BLKp{letter}'],
        raw_test_data[f'TOp{letter}'],
    )
    baseline = baseline.reindex(X_test.index)

    # Test model
    mae = mean_absolute_error(Y_test, model.predict(X_test))
    baseline_average = mean_absolute_error(Y_test, baseline)
    improvement = 100 * (baseline_average - mae) / baseline_average
    print("\n\nMAE                            : {}".format(mae))
    print("Baseline from average DK points: {}".format(baseline_average))
    print("Improvement over baseline      : {} %\n\n".format(round(improvement, 2)))
    import IPython; IPython.embed()

    return model, data_pipeline


def get_regression_model(X, Y, eliminate_keys=False, improvement_percent_gate=0.005):

    #from xgboost import XGBRegressor
    #model = XGBRegressor(objective="reg:squarederror", n_estimators=500)
    #model.fit(X, Y)
    #return model

    ridge_reg = Ridge()
    ridge_reg.fit(X, Y)
    if not eliminate_keys:
        return ridge_reg

    # Iterate until the best final combination of keys is found
    final_keys = []
    final_score = 1000000000
    final_test_ridge_reg = Ridge()
    while True:
        # Look for the next best key
        best_score = 1000000000
        best_key = None
        test_ridge_reg = Ridge()
        # Consider all keys not already in final_keys
        for key in list(set(X.keys()) - set(final_keys)):
            keys = final_keys + [key]
            scores = cross_val_score(
                test_ridge_reg, X[keys], Y, scoring="neg_mean_absolute_error", cv=10
            )
            score = -np.mean(scores)
            if score < best_score:
                best_score = score
                best_key = key
        # If new result is better than the previously found best result, update
        if best_score < final_score and 100*(final_score - best_score) / best_score > improvement_percent_gate:
            final_score = best_score
            final_keys.append(best_key)
            final_test_ridge_reg.fit(X[final_keys], Y)
            print("\nIntermediate state, score: {}:".format(final_score))
            for final_key_i, final_key in enumerate(final_keys):
                print("{}: {}".format(final_key, final_test_ridge_reg.coef_[final_key_i]))
        # Else we are no longer improving, and stop iterating
        else:
            break

    # Remove the keys that are found to not be used
    for key_i, key in enumerate(X.keys()):
        # If key isn't used, set it to 0.0
        if key not in final_keys:
            ridge_reg.coef_[key_i] = 0.0
        # Else key is used, set the coefficient based on the model with the proper keys
        else:
            ridge_reg.coef_[key_i] = final_test_ridge_reg.coef_[final_keys.index(key)]
    ridge_reg.intercept_ = final_test_ridge_reg.intercept_

    return ridge_reg


def print_model(model):
    if not hasattr(model, 'coef_'):
        return
    print("\nModel description:")
    sorted_coef = sorted(enumerate(model.coef_), key=lambda vals: np.abs(vals[1]), reverse=True)
    for feature_i, coef in sorted_coef:
        print('  {:20s} {}'.format(model.feature_names_in_[feature_i], coef))


def get_neural_net_model(X_train, Y_train, X_valid, Y_valid, scan=True):
    if scan:
        return _get_scanned_neural_net_model(X_train, Y_train, X_valid, Y_valid)
    else:
        return _get_basic_neural_net_model(X_train, Y_train, X_valid, Y_valid)


def _get_scanned_neural_net_model(X_train, Y_train, X_valid, Y_valid):

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
            epochs=1000,
            validation_data=(x_val, y_val),
            callbacks=[keras.callbacks.TerminateOnNaN(), early_stopping_cb],
        )
        return out, model

    p = {'n_neurons': list(np.arange(10, 200, 20)) + list(np.arange(200, 500, 50)),
         'n_hidden': [1, 2, 3, 4, 5],
         'activation_fn': ['relu', 'tanh'],
         'neuron_decay': [1.0, 0.8, 0.6, 0.4],
         'sequential': [True, False],
         'learning_rate': [0.001, 0.01, 0.1],
         'patience': [10],
         'batch_size': list(np.arange(10, 240, 30)),
    }

    scan = talos.Scan(
        x=X_train,
        y=Y_train,
        model=build_model,
        params=p,
        experiment_name='test',
        x_val=X_valid,
        y_val=Y_valid,
        val_split=0.2,
        print_params=True,
        #time_limit='2022-1-29 22:15',
        fraction_limit=0.05,
    )

    import IPython; IPython.embed()
    best_model = scan.best_model('val_loss')
    return best_model
    #with open('data.pickle', 'wb') as handle:
    #    pickle.dump(scan.data, handle)

def _get_basic_neural_net_model(X_train, Y_train, X_valid, Y_valid):

    # Create the model's layers
    input_ = keras.layers.Input(shape=X_train.shape[1:])
    hidden1 = keras.layers.Dense(150, activation="relu")(input_)
    hidden2 = keras.layers.Dense(150, activation="relu")(hidden1)
    concat = keras.layers.Concatenate()([input_, hidden2])
    output = keras.layers.Dense(1)(concat)
    model = keras.Model(inputs=[input_], outputs=[output])

    # Train the model
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

    return model


def predict_from_model(model, data_pipeline, data):

    data_X, _, data_accounting, _ = cleanup_data(data, data_pipeline=data_pipeline, train=False)
    if len(data_accounting) == 0:
        return {}
    predictions = model.predict(data_X.loc[data_accounting.index])
    results = {}
    for i, (_, row) in enumerate(data_accounting.iterrows()):
        # Handle whether model returned float or numpy array with 1 float
        results[row.PLAYER_ID] = predictions[i] if isinstance(predictions[i], float) else predictions[i][0]
    return results
