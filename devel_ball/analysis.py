import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from tensorflow import keras
import talos
import pickle
from keras.models import Model, Sequential
from keras.layers import GRU, Dense, LSTM
from keras.callbacks import EarlyStopping

from devel_ball.models import Player


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


def cleanup_data(data, prediction_type=PredictionType.DKPG):
    """
    TODO: docx
    """

    # Shuffle the data such that it is not in order
    #data = data.sample(frac=1)

    # Remove non-players that we don't want to predict based on
    data = data[data['MINpg'] > 10.]
    data = data[data['MIN'] > 0.0]

    # Create lists of relevant categories
    pg_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pg'] # per-game
    pm_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pm'] # per-minute
    pp_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pp'] # per-possession
    rec_cats = [cat for cat in list(data.columns) if cat[:6] == 'RECENT'] # recent-cats
    not_rec_cats = [cat for cat in list(data.columns) if cat[:6] != 'RECENT'] # not-recent-cats

    # List of categories used for predictions
    predictions = ["DK_POINTS", "MIN", "POSS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS"]

    # Create dataframes pertaining to which type of data to predict from
    DKPG_data = data.copy(deep=True) # draftkings points per game prediction
    for cat in pm_cats + pp_cats + rec_cats:
        DKPG_data.pop(cat)
    DKPM_data = data.copy(deep=True) # draftkings points per minute prediction
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
    full_pipeline = get_cleanup_pipeline(data_X)
    data_X_prepared_np = full_pipeline.fit_transform(data_X)
    data_X_prepared = pd.DataFrame(data_X_prepared_np, data_X.index, data_X.columns)
    #data_X_prepared = data_X_prepared[
    #    ['PTSpg', 'FG3Mpg', 'OREBpg', 'DREBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'MINpg', 'POSSpg']
    #]

    return data_X_prepared, data_Y, data_accounting


    ### TEMP - LEFTOVER NOTES FROM TESTING RECENCY DATA ON MINS IT SEEMS ###
    # Create dataframe pertaining to recent data
    #REC_data = data.copy(deep=True)
    #for cat in [c for c in not_rec_cats if c not in predictions]:
    #    REC_data.pop(cat)
    #for cat in list(REC_data.columns):
    #    if 'MIN' not in cat:
    #        REC_data.pop(cat)
    #REC_data.pop('DK_POINTS_PER_MIN')
    # Clean up None's and Nans
    #for i in range(1, 10):
    #    REC_data['RECENT_MIN{}'.format(i)] = (
    #        np.where(REC_data['RECENT_MIN{}'.format(i)].isnull(),
    #            REC_data['RECENT_MIN{}'.format(i-1)],
    #            REC_data['RECENT_MIN{}'.format(i)]
    #        )
    #    )
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
    """
    Currently just a copy and paste of various analysis/model-creation snippets
    """

    data_X, data_Y, data_accounting = cleanup_data(data, PredictionType.DKPG)

    # Create model
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(data_X, data_Y, train_size=0.85)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train_full, Y_train_full, train_size=0.70/0.85)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train_full, Y_train_full)
    mae = mean_absolute_error(Y_test, lin_reg.predict(X_test))
    print(mae)

    return lin_reg



    """
    test_data = pd.read_pickle('[\'2020-21\'].p')
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

def predict_from_model(model, data, date):
    """
    TODO: docx
    """
    data_X, data_Y, data_accounting = cleanup_data(data)
    date_data = data_accounting[data_accounting['DATE'] == date]
    predictions = model.predict(data_X.loc[date_data.index])
    for i, (_, row) in enumerate(date_data.iterrows()):
        print(Player.objects(unique_id=row.PLAYER_ID)[0].name)
        print(predictions[i])
        print()
