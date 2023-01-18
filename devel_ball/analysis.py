import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum
from copy import deepcopy
import itertools

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
from devel_ball.data_transformers import get_cleanup_pipeline


############################################ PARAMATERS ############################################

PARAMS = {

    # Type of predictions to make
    'prediction_stat': 'PTS',
    'prediction_type': 'PM',

    # Thresholds to include players in the data
    'min_MPG': 15.,
    'min_games_played': 5,

    # List of categories used for predictions
    'prediction_categories': [
        "DK_POINTS", "MIN", "POSS", "DK_POINTS_PER_MIN", "DK_POINTS_PER_POSS",
        "PTS", "REB", "AST", "STL", "BLK", "FG3M", "TO"
    ],

    # The ratio of train/test data
    'train_to_test_ratio': 0.85,
    # The ration of train to validation data
    'train_to_validation_ratio': 0.85,

    # Whether or not to make the machine learning random or be seeded for deterministic results
    'random': False,

    # Parameters of the specific machine learning model
    'model': {
        'type': 'RIDGE_REGRESSION',
        'eliminate_keys': False,
        'improvement_percent_gate': 0.005,
    },

    # 'model': {
    #     'type': 'NEURAL_NET',
    #     'scan': False,
    # },

}

####################################################################################################


class PredictionStat(Enum):
    DK_POINTS = 1
    MIN = 2
    POSS = 3
    PTS = 4
    REB = 5
    AST = 6
    FG3M = 7
    BLK = 8
    STL = 9
    TO = 10


class PredictionType(Enum):
    PG = 1  # per game
    PM = 2  # per minute
    PP = 3  # per possession


class ModelType(Enum):
    RIDGE_REGRESSION = 1
    NEURAL_NET = 2


def get_categories(data):
    """
    Extract the categories from the dataset.

    :returns:
        per game cateogires,
        per minute categories,
        per possession categories,
        recent categories,
        accounting categories,
    """
    pg_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pg'] # per-game
    pm_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pm'] # per-minute
    pp_cats = [cat for cat in list(data.columns) if cat[-2:] == 'pp'] # per-possession
    rec_cats = [cat for cat in list(data.columns) if cat[:6] == 'RECENT'] # recent-cats
    accounting_cats = ['PLAYER_ID', 'DATE']
    return pg_cats, pm_cats, pp_cats, rec_cats, accounting_cats


def cleanup_recent_cats(data, rec_cats, prediction_type):
    """
    Cleanup the recent categories, where data may be missing (like if you've only played 2 games so far,
    yet we track for up to 10 games.

    This is handled by averaging the recent game data that we do have, and applying this backwards for
    the rest of the untracked games. So if you've only played 2 games and you had 10 and 20 PTS, then
    the previous 8 games before this would be marked as 15 PTS.

    Also, update the recent categories for the specified prediction type. For example if the
    prediction type is per minute, then update the recent categories to be recent per minute stats,
    i.e. points per minute.

    :param data: The input data to cleanup
    :param rec_cats: the recent categories in the dataset
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    """

    # Figure out all the unique recent categories, and how far back they go in "lag" of games
    unique_recent_cats = set([c[:-1] for c in rec_cats])
    random_recent_cat = list(unique_recent_cats)[0]
    max_lag = max(
        int(c[len(random_recent_cat):]) for c in rec_cats if random_recent_cat in c
    )

    # Update the recent stats that are not filled out with the average up until that point
    for rec_cat in unique_recent_cats:
        for i in range(1, max_lag + 1):
            data['{}{}'.format(rec_cat, i)] = (
                np.where(
                    data['{}{}'.format(rec_cat, i)].isnull(),
                    sum(data['{}{}'.format(rec_cat, j)] for j in range(i))/i,
                    data['{}{}'.format(rec_cat, i)],
                )
            )

    # Update the recent categories for the given prediction type if per minute or per possession.
    cats_to_update = deepcopy(unique_recent_cats)
    if prediction_type == PredictionType.PG:
        cats_to_update = []
    elif prediction_type == PredictionType.PM:
        divisor_cat = 'RECENT_MIN'
        suffix = 'pm'
        # If predicting based on per minute stats, don't update the recent minutes category
        cats_to_update.remove('RECENT_MIN')
    elif prediction_type == PredictionType.PP:
        divisor_cat = 'RECENT_POSS'
        suffix = 'pp'
        # If predicting based on per possession stats, don't update the recent possessions category
        cats_to_update.remove('RECENT_POSS')
    else:
        raise Exception("Invalid specified PredictionType: {}".format(prediction_type))
    for rec_cat in cats_to_update:
        for i in range(max_lag + 1):
            new_cat = (
                data['{}{}'.format(rec_cat, i)].astype('float64')
                    / data['{}{}'.format(divisor_cat, i)].astype('float64')
            ).replace((np.inf, -np.inf, np.nan), (0.0, 0.0, 0.0))
            # Add the new category
            data['{}{}{}'.format(rec_cat, i, suffix)] = new_cat
            # Remove the old category
            data.pop('{}{}'.format(rec_cat, i))

    # Replace the raw recent data with weighted averages, don't update RECENT_MIN though for
    # pm predictions, or RECENT_POSS for pp predictions
    weighted_average_cats = deepcopy(unique_recent_cats)
    suffix = ''
    check_for_playing_category = 'MIN'
    if prediction_type == PredictionType.PM:
        weighted_average_cats.remove('RECENT_MIN')
        suffix = 'pm'
        check_for_playing_category = 'MIN'
    elif prediction_type == PredictionType.PP:
        weighted_average_cats.remove('RECENT_POSS')
        suffix = 'pp'
        check_for_playing_category = 'POSS'

    # When creating the new weighted averages, consider that a player may miss games. Rather than giving him
    # a "0" for this value, we fill in the gaps by just not considering the missed game as a part of the weighted
    # average calculation
    for data_index, row in data.iterrows():
        for recent_cat in weighted_average_cats:
            numerator = denominator = 0
            for i in range(max_lag+1):
                if row['RECENT_{}{}'.format(check_for_playing_category, i)] > 0.0:
                    numerator += row['{}{}{}'.format(recent_cat, i, suffix)] * (max_lag + 1 - i)
                    denominator += (max_lag + 1 - i)
            data.at[data_index, '{}{}_weighted_average'.format(recent_cat, suffix)] = numerator/denominator
    # Remove the old raw recent stats that are of no interest now that the weighted average exists
    for recent_cat in weighted_average_cats:
        for i in range(max_lag+1):
            data.pop('{}{}{}'.format(recent_cat, i, suffix))

    return data


def filter_data_by_prediction_type(data, prediction_type, pg_cats, pm_cats, pp_cats):
    """
    Take the data, and filter is based on the specified prediction type. This removes the irrelevant
    categories, for example if predicting per game, remove the per minute and per possession stats

    :param data: The input data to cleanup
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    :param pg_cats: the per game categories in the dataset
    :param pm_cats: the per minute categories in the dataset
    :param pp_cats: the per possession categories in the dataset
    """

    # Remove categories that don't matter based on prediction type
    remove_cats = {
        PredictionType.PG: pm_cats + pp_cats,
        PredictionType.PM: pg_cats + pp_cats,
        PredictionType.PP: pg_cats + pm_cats,
    }[prediction_type]
    for cat in remove_cats:
        data.pop(cat)

    return data


def filter_data_for_accounting_data(data, accounting_cats):
    """
    Filter the accounting cats out of the data, and then return it as a new dataset.

    :param data: The input data to cleanup
    :param account_cats: the accounting categories
    """
    data_accounting = data[accounting_cats].copy()
    data = data.drop(accounting_cats, axis=1)
    return data, data_accounting


def separate_predict_and_results_data(data, prediction_stat, prediction_type, prediction_cats):
    """
    Take the full data set, and separate it into data to use to predict with, and the results
    data that is what can be checked against.

    Also convert the data to predict to the right type of category, i.e. pm or pp.

    :param data: The input data to cleanup
    :param prediction_stat: The stat that will be predicted (DK_POINTS, PTS, REBS, ...)
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    :param prediction_cats: The categories that can be predicted, representing the results

    :returns: the data that can be used to predict, and the data that can be predicted
    """

    # Separate based on prediction categories
    data_X = data.drop(prediction_cats, axis=1)

    # Figure out the categories to keep
    if prediction_type == PredictionType.PG:
        data_Y = data[[prediction_stat.name]].astype('float64')
    elif prediction_type == PredictionType.PM:
        data_Y = data[['MIN']].astype('float64')
        data_Y['{}pm'.format(prediction_stat.name)] = (
            data[prediction_stat.name].astype('float64') / data['MIN'].astype('float64')
        ).replace((np.inf, -np.inf, np.nan), (0.0, 0.0, 0.0))
    elif prediction_type == PredictionType.PP:
        data_Y = data[['POSS']].astype('float64')
        data_Y['{}pp'.format(prediction_stat.name)] = (
            data[prediction_stat.name].astype('float64') / data['POSS'].astype('float64')
        ).replace((np.inf, -np.inf, np.nan), (0.0, 0.0, 0.0))
    else:
        raise Exception("Invalid specified PredictionType: {}".format(prediction_type))

    return data_X, data_Y.astype('float64')


def add_features(data, prediction_stat, prediction_type):
    """
    Add any new features to the model for predictions.

    1) If prediction draftkings points, add draftkings points per game
    2) Add game stats, which are averages of both teams individual stats in the game. Currently, we still leave
       in the individual team stats (home and away) and just add the extra stat, without removing these.

    :param data: The input data to cleanup
    :param prediction_stat: The stat that will be predicted (DK_POINTS, PTS, REBS, ...)
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    """
    # If predicting draftkings, add draftkings points per game
    if prediction_stat == PredictionStat.DK_POINTS:
        DK_POINTSpg = get_dk_points(
            data['PTSpg'],
            data['FG3Mpg'],
            data['REBpg'],
            data['ASTpg'],
            data['STLpg'],
            data['BLKpg'],
            data['TOpg'],
        )
        data['DK_POINTSpg'] = DK_POINTSpg.astype('float64')

    # Add "game stats" which are cumulative of the 2 teams, currently we still leave in the individual team
    # stats too in case those are relevant as well
    for team_stat_category in [cat[:-2] for cat in list(data.columns) if cat[-2:] == 'tm']:
        game_stat = (
            data['{}{}'.format(team_stat_category, 'tm')] + data['{}{}'.format(team_stat_category, 'vsTm')]
        ) / 2.0
        data['{}{}'.format(team_stat_category, 'gm')] = game_stat

    return data


def cleanup_data(
    data,
    prediction_stat,
    prediction_type,
    prediction_cats,
    data_pipeline=None,
    train=True,
    min_MPG=15.,
    min_games_played=5,
):
    """
    Prepare the raw input data for usage by the model. Create the data_pipeline if one is provided.

    :param data: The input data to cleanup
    :param prediction_stat: The stat that will be predicted (DK_POINTS, PTS, REBS, ...)
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    :param prediction_cats: The categories that could possibly be predicted
    :param data_pipline: If provided, use this as the data cleanup pipeline, if not provided, then
        create the data cleanup pieline
    :param bool train: Whether or not this is a part of training
    :param min_MPG: the minimum MPG to include a player in the data
    :param min_games_played: the minimum games played thus far this year to include the player in the data

    :returns: data_X, data_Y, data_accounting, data_pipeline
    """

    # Filter out players who haven't played the adequate amount of minute of games
    data = data[data['MINpg'] >=  min_MPG]
    try:
        data = data[data['RECENT_PTS{}'.format(min_games_played-1)].notnull()]
    except KeyError:
        # The min number of games played exceeds the amount we track, invalid state
        print(
            "The min_games_played of {} is more than the amount we track, this should be lowered."
            .format(min_games_played)
        )

    if train:
        # For training, we'll also remove players who didn't play in the game, and assume we would have known
        # that someone who isn't playing (likely injured) wouldn't have played anyways. Also, let's exclude
        # players who haven't played 5 games yet this season.
        data = data[data['MIN'] > 0.0]

    # Separate the categories into their given types
    pg_cats, pm_cats, pp_cats, rec_cats, accounting_cats = get_categories(data)

    # Cleanup the recent categories where data may be missing, and update for specified prediction_type
    data = cleanup_recent_cats(data, rec_cats, prediction_type)

    # Filter the data
    data = filter_data_by_prediction_type(data, prediction_type, pg_cats, pm_cats, pp_cats)
    data, data_accounting = filter_data_for_accounting_data(data, accounting_cats)

    # Separate prediction data from results data, and match the results data to the desired prediction
    # stat and type
    data_X, data_Y = separate_predict_and_results_data(
        data, prediction_stat, prediction_type, prediction_cats,
    )

    # Add new features to potentially make use of
    data_X = add_features(data_X, prediction_stat, prediction_type)

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


def filter_data_for_ml_model(data_X, data_Y, prediction_stat, prediction_type):
    """
    Extract the components of data_X and data_Y relevant to the machine learning model. For example, if
    predicting PTS and PM (per minute), then data_X would be filtered to only include the data relevant to
    predictiong PTS, and data_Y would be filtered to only PTS.

    :param data_X: the prediction data
    :param data_Y: the results data
    :param prediction_stat: The stat that will be predicted (DK_POINTS, PTS, REBS, ...)
    :param prediction_type: The type of prediction to make (per game, per minute, per possession)
    """

    # Make data_Y only have the prediction_stat, and remove any denominator data like MIN or POSS
    suffix = {
        PredictionType.PG: '',
        PredictionType.PM: 'pm',
        PredictionType.PP: 'pp',
    }[prediction_type]
    data_Y = data_Y[["{}{}".format(prediction_stat.name, suffix)]]

    # If prediction PM or PP, then filter out the recent MIN or POSS data that was stored but not used
    # as a part of the machine learning model.
    remove_cats = []
    if prediction_type == PredictionType.PM:
        remove_cats.extend(c for c in data_X.columns if 'RECENT_MIN' in c)
    elif prediction_type == PredictionType.PP:
        remove_cats.extend(c for c in data_X.columns if 'RECENT_POSS' in c)
    for remove_cat in remove_cats:
        data_X.pop(remove_cat)

    return data_X, data_Y


def get_regression_model(
    X_train, Y_train, X_validation, Y_validation, random, eliminate_keys=False, improvement_percent_gate=0.005
):

    # Because this model explicitly uses direct training, or cross validation which handles train/validation
    # splitting, we might as well use all the data we have available
    X_train = X_train.append(X_validation)
    X_train = X_train.reset_index()
    X_train = X_train.drop('index', axis=1)
    Y_train = Y_train.append(Y_validation)
    Y_train = Y_train.reset_index()
    Y_train = Y_train.drop('index', axis=1)

    def print_model(model):
        sorted_coef = sorted(enumerate(model.coef_[0]), key=lambda vals: np.abs(vals[1]), reverse=True)
        for feature_i, coef in sorted_coef:
            print('  {:35s} {}'.format(model.feature_names_in_[feature_i], round(coef, 6)))

    ridge_reg = Ridge()
    ridge_reg.fit(X_train, Y_train)
    if not eliminate_keys:
        print("\nModel description:")
        print_model(ridge_reg)
        return ridge_reg

    ### TODO (JS): test better elimination of keys ###
    # TODO (JS): probably want to make this an enum of like type "ALL", "ADD_KEYS", "REMOVE_KEYS". probably something
    # worth being able to test/iterate on differently for different stats/pg/pm/pp

    final_keys = set(X_train.keys())
    final_score = -np.mean(
        cross_val_score(ridge_reg, X_train, Y_train, scoring="neg_mean_absolute_error", cv=10)
    )
    final_test_ridge_reg = Ridge()
    print("\nIteratively removing keys to improve the model. Starting score: {}".format(final_score))
    while True:

        # TODO (JS): this is for testing:
        print("total keys remaining: {}".format(len(final_keys)))

        # Look for the next best key to remove
        best_score = np.inf
        best_key_to_remove = None
        test_ridge_reg = Ridge()
        # Consider all keys still existing in final keys:
        for key_to_test in final_keys:
            keys = [key for key in final_keys if key != key_to_test]
            score = -np.mean(
                cross_val_score(
                    test_ridge_reg, X_train[keys], Y_train, scoring="neg_mean_absolute_error", cv=10
                )
            )
            if score < best_score:
                best_score = score
                best_key_to_remove = key_to_test
        # If new result is better than the previously found best result, update
        # TODO (JS): figure out if improvement gate makes sense here, may need tuning
        if best_score < final_score:
            final_score = best_score
            final_keys.remove(best_key_to_remove)
            final_test_ridge_reg.fit(X_train[final_keys], Y_train)
            # TODO (JS): format this print
            print("\nIntermediate state, removed: {}, score: {}:".format(best_key_to_remove, final_score))
        # Else we are no longer improving, and stop iterating
        else:
            break

    # TODO (JS): de-dupe this logic in both methods of doing key removal/addition
    # Remove the keys that are found to not be used
    import IPython; IPython.embed()
    for key_i, key in enumerate(X_train.keys()):
        # If key isn't used, set it to 0.0
        if key not in final_keys:
            ridge_reg.coef_[0][key_i] = 0.0
        # Else key is used, set the coefficient based on the model with the proper keys
        else:
            ridge_reg.coef_[0][key_i] = final_test_ridge_reg.coef_[0][list(final_keys).index(key)]
    ridge_reg.intercept_ = final_test_ridge_reg.intercept_

    print("\nFinal Model description:")
    print_model(ridge_reg)

    return ridge_reg



    ###################################################

    # Iterate until the best final combination of keys is found
    print("\nIteratively adding keys to improve the model.")
    final_keys = []
    final_score = np.inf
    final_test_ridge_reg = Ridge()
    while True:
        # Look for the next best key
        best_score = np.inf
        best_key = None
        test_ridge_reg = Ridge()
        # Consider all keys not already in final_keys
        for key in list(set(X_train.keys()) - set(final_keys)):
            keys = final_keys + [key]
            score = -np.mean(
                cross_val_score(
                    test_ridge_reg, X_train[keys], Y_train, scoring="neg_mean_absolute_error", cv=10
                )
            )
            if score < best_score:
                best_score = score
                best_key = key
        # If new result is better than the previously found best result, update
        if best_score < final_score and 100*(final_score - best_score) / best_score > improvement_percent_gate:
            final_score = best_score
            final_keys.append(best_key)
            final_test_ridge_reg.fit(X_train[final_keys], Y_train)
            print("\nIntermediate state, score: {}:".format(final_score))
            print_model(final_test_ridge_reg)
        # Else we are no longer improving, and stop iterating
        else:
            break

    # Remove the keys that are found to not be used
    for key_i, key in enumerate(X_train.keys()):
        # If key isn't used, set it to 0.0
        if key not in final_keys:
            ridge_reg.coef_[0][key_i] = 0.0
        # Else key is used, set the coefficient based on the model with the proper keys
        else:
            ridge_reg.coef_[0][key_i] = final_test_ridge_reg.coef_[0][final_keys.index(key)]
    ridge_reg.intercept_ = final_test_ridge_reg.intercept_

    print("\nFinal Model description:")
    print_model(ridge_reg)

    return ridge_reg


def _get_scanned_neural_net_model(X_train, Y_train, X_validation, Y_validation, **ignored_kwargs):

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
         'patience': [100],
         'batch_size': list(np.arange(10, 240, 30)),
    }

    scan = talos.Scan(
        x=X_train,
        y=Y_train,
        model=build_model,
        params=p,
        experiment_name='test',
        x_val=X_validation,
        y_val=Y_validation,
        val_split=0.2,
        print_params=True,
        #time_limit='2022-1-29 22:15',
        fraction_limit=0.05,
    )

    best_model = scan.best_model('val_loss')
    return best_model
    #with open('data.pickle', 'wb') as handle:
    #    pickle.dump(scan.data, handle)


def _get_basic_neural_net_model(X_train, Y_train, X_validation, Y_validation, **ignored_kwargs):

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
    early_stopping_cb = keras.callbacks.EarlyStopping(patience=100, restore_best_weights=True)
    history = model.fit(
        X_train,
        Y_train,
        epochs=200,
        batch_size=80,
        validation_data=(X_validation, Y_validation),
        callbacks=[early_stopping_cb],
    )

    return model


def get_neural_net_model(
    X_train, Y_train, X_validation, Y_validation, random, train_to_valid_ratio=0.85, scan=False, **params
):
    model_creator = _get_scanned_neural_net_model if scan else _get_basic_neural_net_model
    return model_creator(X_train, Y_train, X_validation, Y_validation, **params)


def get_model(data):

    # Pull the "parameters"
    min_MPG = PARAMS['min_MPG']
    min_games_played = PARAMS['min_games_played']
    prediction_stat = PredictionStat[PARAMS['prediction_stat']]
    prediction_type = PredictionType[PARAMS['prediction_type']]
    prediction_cats = PARAMS['prediction_categories']
    train_to_test_ratio = PARAMS['train_to_test_ratio']
    train_to_validation_ratio = PARAMS['train_to_validation_ratio']
    random = PARAMS['random']
    model_params = PARAMS['model']

    # Get the cleaned data, as well as the data pipeline to use to clean it. The return values here include
    # all the possible prediction data in data_X, and then all the things that can be predicted in data_Y, for
    # example if predictiong PTS and PM (per minute), then data_Y includes MIN and PTSpm.
    data_X, data_Y, data_accounting, data_pipeline = cleanup_data(
        data,
        prediction_stat=prediction_stat,
        prediction_type=prediction_type,
        prediction_cats=prediction_cats,
        train=True,
        min_MPG=min_MPG,
        min_games_played=min_games_played,
    )

    """
    data_X = data_X[[
        'PTSpm',
        'AST_TOVgm',
        'RECENT_PTSpm_weighted_average',
        'PER',
        'PIEgm',
        'DREBpm',
        'TOvsTm',
        'FG3Mpm',
        'REB_PCT',
        'TOpm',
        'FG3Apm',
        'TS_PCT',
        'BLKgm',
        'PIEtm',
        'PIE',
        'TOtm',
        'AST_PCTvsTm',
        'PACEtm',
        'AST_TOV',
        'BLKtm',
        'RECENT_USG_PCTpm_weighted_average',
        'GAME_SCORE',
        'AST_PCTtm',
        'BLKvsTm',
        'POSStm',
        'OREBpm',
        'RECENT_TOpm_weighted_average',
        'PTStm',
        'RECENT_ASTpm_weighted_average',
        'PLUS_MINUSpm',
        'AST_PCT',
        'STLtm',
        'BLKpm',
        'PFtm',
        'FTP',
        'FG3Ptm',
        'FTPtm',
        'FTPvsTm',
        'DEF_RTG',
        'DREBvsTm',
        'HOME',
    ]]
    """

    # Extract the components of data_X and data_Y relevant to the machine learning model. For example, if
    # predicting PTS and PM (per minute), then data_X would be filtered to only include the data relevant to
    # predictiong PTS, and data_Y would be filtered to only PTS.
    data_X, data_Y = filter_data_for_ml_model(data_X, data_Y, prediction_stat, prediction_type)

    # Split data into train and test data
    X_train_full, X_test, Y_train_full, Y_test = train_test_split(
        data_X,
        data_Y,
        train_size=train_to_test_ratio,
        shuffle=True,
        random_state=None if random else 42,
    )

    # Split the train data into train and validation data
    X_train, X_validation, Y_train, Y_validation = train_test_split(
        X_train_full,
        Y_train_full,
        train_size=train_to_validation_ratio,
        shuffle=True,
        random_state=None if random else 42,
    )

    # Create the model depending on specified type
    model_creator = {
        ModelType.RIDGE_REGRESSION: get_regression_model,
        ModelType.NEURAL_NET: get_neural_net_model,
    }[ModelType[model_params['type']]]
    print("Creating model of type: {}\n".format(model_params['type']))
    model_params.pop('type')
    model = model_creator(X_train, Y_train, X_validation, Y_validation, random=random, **model_params)

    # Get baseline by just looking at the given average for the category
    raw_test_data = data[data.index.isin(X_test.index)]
    # Handle draft kings separately here
    if prediction_stat == PredictionStat.DK_POINTS:
        baseline = get_dk_points(
            raw_test_data['PTSpg'],
            raw_test_data['FG3Mpg'],
            raw_test_data['REBpg'],
            raw_test_data['ASTpg'],
            raw_test_data['STLpg'],
            raw_test_data['BLKpg'],
            raw_test_data['TOpg'],
        )
        stat_name = "DK_POINTSpg"
        if prediction_type == PredictionType.PM:
            baseline /= raw_test_data['MINpg']
            stat_name = "DK_POINTSpm"
        elif prediction_type == PredictionType.PP:
            baseline /= raw_test_data['POSSpg']
            stat_name = "DK_POINTSpp"
    else:
        suffix = {
            PredictionType.PG: 'pg',
            PredictionType.PM: 'pm',
            PredictionType.PP: 'pp',
        }[prediction_type]
        stat_name = "{}{}".format(prediction_stat.name, suffix)
        baseline = raw_test_data[stat_name]
    baseline = baseline.reindex(X_test.index)

    # Test model
    mae = mean_absolute_error(Y_test, model.predict(X_test))
    baseline_average = mean_absolute_error(Y_test, baseline)
    improvement = 100 * (baseline_average - mae) / baseline_average
    print('\n\n{:35s}                 : {}'.format("MAE", mae))
    print('{:35s}                 : {}'.format("Baseline from average {}".format(stat_name), baseline_average))
    print('{:35s}                 : {} %\n\n'.format("Improvement over baseline", round(improvement, 2)))

    # import IPython; IPython.embed()
    return model, data_pipeline


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
