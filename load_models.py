#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
import datetime
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd
from json import JSONDecodeError

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    scoreboardv2,
    boxscoresummaryv2,
    boxscoretraditionalv2,
    boxscoreadvancedv2,
    boxscoreusagev2,
)

from models import (
    Player,
    Team,
    GameDate,
    Season,
    Game,
    Official,
    PlayerGame,
    TeamGame,
    GameTraditionalStats,
    GameAdvancedStats,
    GameUsageStats,
)


def query_nba_api(endpoint, sleep_time=1, quiet=False, **kwargs):
    """
    Query the nba_api at a safe rate

    :param endpoint: the nba_api endpoint to utilize
    :type  endpoint: function
    :param sleep_time: amount to sleep before lookup to not overload the server
    :type  sleep_time: int
    """
    if not quiet:
        print("Querying nba_api {} with args: {}".format(
            str(endpoint).split(".")[-1].split('\'')[0],
            kwargs)
        )
    sleep(sleep_time)
    return endpoint(**kwargs)


def add_entry_to_db(document_type, unique_id, name, year, game_id):
    """
    Add Player, Team, or Official entry to database if doesn't exist already.
    Add the supplied year if it doesn't exist already, and then add the game_id
    to that year's list of games.

    :param document_type: Player, Team, or Official
    :type  document_type: Mongo Document
    :param unique_id: player, team, or official's unique id
    :type  unique_id: str
    :param name: player's full name, team's full name, or official full name
    :type  name: str
    :param year: year that was played in
    :type  year: str
    :param game_id: id of game
    :type  game_id: str
    """

    # Fetch entry if it exists, or create it if it doesn't alrady
    entry = document_type.objects(unique_id=unique_id)
    if entry:
        entry = entry[0]
    else:
        entry = document_type(unique_id=unique_id)
        entry.name = name

    # Add year to list of years if it doesn't exist yet
    if year not in entry.years:
        entry.years[year] = []

    # Append game_id to list of games for year
    entry.years[year].append(game_id)

    # Save updated entry
    entry.save()
    return entry


def clean_boxscore_df(df, index, str_keys=['PLAYER_ID', 'TEAM_ID']):
    """
    Clean a dataframe by:
    - converting str_keys columns to string
    - converting all NaNs to 0.0
    - updating MIN field
    - setting the index

    :param df: dataframe to clean
    :type  df: pandas.dataframe
    :param index: which key to use as index
    :type  index: str
    :param str_keys: which columns to convert to strings
    :type  str_keys: [str]
    """

    # Convert relevant fields to strings
    for key in str_keys:
        try:
            df[key] = df[key].apply(lambda value: str(value))
        except KeyError:
            pass
    # Update min field from (min:sec) to (minutes) as float
    try:
        df['MIN'] = df['MIN'].apply(lambda MIN:
            np.round(float(MIN.split(':')[0]) + float(MIN.split(':')[1])/60.0, 2) if MIN else 0.0)
    except KeyError:
        pass
    # Update all NaNs to 0.0
    df = df.fillna(0.0)
    # Convert ints to floats
    df = df.apply(lambda x: x.astype('float64') if x.name in
                  df.select_dtypes(np.integer).keys() else x)
    # Set index of df and return
    return df.set_index(index)


def assign_all_values(mongo_entry, df):
    """
    Assign all values for each key that exist in mongo_entry to the same keyed value in df

    :param mongo_entry: entry in mongo to store data
    :type  mongo_entry: GameTraditionalStats
    :param df: dataframe which has at least each identical key taht exists in mongo_entry
    :type  df: pandas.dataframe
    """
    for key in mongo_entry:
        mongo_entry[key] = df[key]


def get_gamedates(years):
    """
    Load the dates and games played on them for the given years supplied

    :param years: the years to load info for
    :type  years: list[str]
    """

    # For each specified year, look at the dates and games played on them
    for year in years:
        season_entry = Season()
        season_entry.year = year

        # Get the first day of October as the first possible default date
        first_date = '{}-10-01'.format(year[:4])

        # Iterate until finding first day of regular season
        while True:
            print("Looking at {} for first day of season".format(first_date))
            gameday = query_nba_api(scoreboardv2.ScoreboardV2, game_date=first_date)
            game_ids = gameday.available.get_data_frame()['GAME_ID']
            # If there were games this day, and it is regular season
            if len(game_ids)>0 and game_ids[0][2]=='2':
                season_entry.first_date = first_date
                break
            else:
                first_date = (
                    datetime.date.fromisoformat(first_date) +
                    timedelta(1)).isoformat()

        # Begin loading into mongo the game dates
        date = first_date
        while True:

            gamedate_entry = GameDate.objects(date=date)

            # Game date already exists in database
            if gamedate_entry:
                print('{} is already in the database'.format(date))
            # Else game date is not already in database
            else:
                gameday = query_nba_api(scoreboardv2.ScoreboardV2, game_date=date)
                game_ids = (
                    gameday.available.get_data_frame()['GAME_ID'].to_list()
                )

                # If all star game, skip
                if len(game_ids)>0 and game_ids[0][2] == '3':
                    game_ids = []
                # If playoff game, stop and mark previous date as last day
                if len(game_ids)>0 and game_ids[0][2] == '4':
                    last_date = (
                        datetime.date.fromisoformat(date) - timedelta(1)
                        ).isoformat()
                    season_entry.last_date = last_date
                    if not Season.objects(year=year):
                        season_entry.save()
                    break

                # Create gameday entry for this day
                gamedate_entry = GameDate()
                gamedate_entry.date = date
                gamedate_entry.year = year
                if '0021201214' in game_ids:  # Remove not played game
                    game_ids.remove('0021201214')
                gamedate_entry.games = game_ids
                gamedate_entry.save()
                print('Adding {} to database with {} games played on '
                      'this day'.format(date, len(game_ids)))

            date = (datetime.date.fromisoformat(date) + timedelta(1)).isoformat()


def get_games(years):
    """
    Load the info and statistics for the game played in the specified years.
    Must have already loaded the gamedates for these years in order to fetch
    the games themselves.

    :param years: the years to load info for
    :type  years: list[str]
    """

    # For each specified year, look at the dates and games played on them
    for year in years:

        # Load season
        season = Season.objects(year=year)
        if not season:
            print('Season and GameDates not yet loaded for {}'.format(year))
            continue
        season=season[0]
        first_date = season.first_date
        last_date = season.last_date

        # For each day in the season
        days = (datetime.date.fromisoformat(last_date) -
                datetime.date.fromisoformat(first_date)).days + 1
        for date in (datetime.date.fromisoformat(first_date) + timedelta(n)
                for n in range(days)):

            print('\n{}     Loading date: {}     {}\n'.format('#'*20, date, '#'*20))
            date=date.isoformat()

            # Fetch GameDate, if it doesn't exist then the season didn't
            # properly load and should re-load it
            game_date = GameDate.objects(date=date)
            if not game_date:
                print('GameDate not loaded for {}, you should re-load this '
                      'season {} in full to get the full season before '
                      'proceeding.'.format(date, year))
                break
            game_date = game_date[0]

            # For each game on this day
            invalid_game_ids = []
            for game_id in game_date.games:

                # Fetch Game, if it exists already, skip it
                game = Game.objects(game_id=game_id)
                if game:
                    continue
                game = Game(game_id=game_id)
                game.date = date
                game.year = year

                # Fetch Box Score Summary
                try:
                    box_score_summary = query_nba_api(
                        boxscoresummaryv2.BoxScoreSummaryV2, game_id=game_id)
                except JSONDecodeError:
                    invalid_game_ids.append(game_id)
                    # The purpose of this except block is because in 2019-20, covid led
                    # to games being cancelled. Fuck COVID.
                    if year == '2019-20':
                        print('Fuck COVID. This game was cancelled.')
                        continue
                    else:
                        raise Exception("Game wasn't found.".format(game_id))

                # Store inactive players
                game.inactives = [
                    str(inactive_player) for inactive_player in
                    box_score_summary.inactive_players.get_data_frame()
                        ['PLAYER_ID'].to_list()
                ]

                # Store officials for this game (create Official if needed)
                officials_df = clean_boxscore_df(
                    box_score_summary.officials.get_data_frame(), index='OFFICIAL_ID',
                    str_keys=['OFFICIAL_ID'])
                officials = []
                game.officials = officials
                for official_id, official in officials_df.iterrows():
                    official_name = '{} {}'.format(official['FIRST_NAME'], official['LAST_NAME'])
                    official_entry = add_entry_to_db(
                        document_type=Official,
                        unique_id=official_id,
                        name=official_name,
                        year=year,
                        game_id=game_id
                    )
                    officials.append(official_id)

                # Store home team id and road team id
                game_summary = box_score_summary.game_summary.get_data_frame()
                home_team_id = str(game_summary['HOME_TEAM_ID'][0])
                road_team_id = str(game_summary['VISITOR_TEAM_ID'][0])

                # Fetch various relevant box scores to use
                # Traditional box score
                box_score_traditional = query_nba_api(
                    boxscoretraditionalv2.BoxScoreTraditionalV2, game_id=game_id)
                players_traditional = clean_boxscore_df(
                    box_score_traditional.player_stats.get_data_frame(), index='PLAYER_ID')
                teams_traditional = clean_boxscore_df(
                    box_score_traditional.team_stats.get_data_frame(), index='TEAM_ID')
                # Advanced box score
                box_score_advanced = query_nba_api(
                    boxscoreadvancedv2.BoxScoreAdvancedV2, game_id=game_id)
                players_advanced = clean_boxscore_df(
                    box_score_advanced.player_stats.get_data_frame(), index='PLAYER_ID')
                teams_advanced = clean_boxscore_df(
                    box_score_advanced.team_stats.get_data_frame(), index='TEAM_ID')
                # Usage box score
                box_score_usage = query_nba_api(
                    boxscoreusagev2.BoxScoreUsageV2, game_id=game_id)
                players_usage = clean_boxscore_df(
                    box_score_usage.sql_players_usage.get_data_frame(), index='PLAYER_ID')

                # Log the current game
                team_names = ['{} {}'.format(team['TEAM_CITY'], team['TEAM_NAME'])
                              for _, team in teams_traditional.iterrows()]
                print('\n{}     Loading game: {} vs. {}     {}'
                      .format('#'*10, team_names[0], team_names[1], '#'*10))

                # Create each PlayerGame and map them to player_id
                player_games = {}
                game.player_games = player_games
                for player_id, player in players_traditional.iterrows():

                    # Gather player info and add to db for this year if not already stored
                    player_name = player['PLAYER_NAME']
                    print("Player: {}  (id: {})".format(player_name, player_id))
                    add_entry_to_db(
                        document_type=Player,
                        unique_id=player_id,
                        name=player_name,
                        year=year,
                        game_id=game_id
                    )

                    # Create PlayerGame entry to add to this game
                    player_game = PlayerGame(player_id=player_id)
                    player_games[player_id] = player_game

                    # Store basic data about PlayerGame
                    if player['TEAM_ID'] == home_team_id:
                        player_game.home = True
                        player_game.team_id = home_team_id
                        player_game.opposing_team_id = road_team_id
                    else:
                        player_game.home = False
                        player_game.team_id = road_team_id
                        player_game.opposing_team_id = home_team_id

                    # Create traditional stats entry for this player
                    traditional_player_entry = GameTraditionalStats()
                    player_game.traditional_stats = traditional_player_entry
                    assign_all_values(traditional_player_entry, player)

                    # Create advanced stats entry for this player
                    advanced_player_entry = GameAdvancedStats()
                    player_game.advanced_stats = advanced_player_entry
                    assign_all_values(advanced_player_entry, players_advanced.loc[player_id])

                    # Create usage stats entry for this player
                    usage_player_entry = GameUsageStats()
                    player_game.usage_stats = usage_player_entry
                    assign_all_values(usage_player_entry, players_usage.loc[player_id])


                # Create each TeamGame and map them to team_id
                team_games = {}
                game.team_games = team_games
                for team_id, team in teams_traditional.iterrows():

                    # Gather team info and add to db for this year if not already stored
                    team_name = '{} {}'.format(team['TEAM_CITY'], team['TEAM_NAME'])
                    print("Team: {}  (id: {})".format(team_name, team_id))
                    add_entry_to_db(
                        document_type=Team,
                        unique_id=team_id,
                        name=team_name,
                        year=year,
                        game_id=game_id
                    )

                    # Create TeamGame entry to add to this game
                    team_game = TeamGame(team_id=team_id)
                    team_games[team_id] = team_game

                    # Store basic data about TeamGame
                    team_game.date = date
                    if team_id == home_team_id:
                        team_game.home = True
                        team_game.opposing_team_id = road_team_id
                    else:
                        team_game.home = False
                        team_game.opposing_team_id = home_team_id

                    # Create traditional stats entry for this team
                    traditional_team_entry = GameTraditionalStats()
                    team_game.traditional_stats = traditional_team_entry
                    assign_all_values(traditional_team_entry, team)

                    # Create advanced stats entry for this team
                    advanced_team_entry = GameAdvancedStats()
                    team_game.advanced_stats = advanced_team_entry
                    assign_all_values(advanced_team_entry, teams_advanced.loc[team_id])

                # Save game
                game.save()
                print("")

            # Remove game_id of games that were cancelled (covid) from game dates for
            # future iterations
            game_date.games = [game_id for game_id in game_date.games if game_id not in invalid_game_ids]
            game_date.save()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--drop', default=[], nargs='*',
                        help='Which collections to drop, can specify \'all\', '
                             '\'all-except-game-dates\', or specific collections.'
                       )
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to acquire player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--gamedates', action='store_true',
                        help='Get dates and games played on those dates.'
                       )
    parser.add_argument('--games', action='store_true',
                        help='Get all info and stats for games played on the '
                             'specified years.  Must have --gamedates already '
                             'fetched for this year.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('devel_ball')

    # Drop specified collections if desired
    drop = []
    if 'all' in args.drop:
        drop = mongo_client['devel_ball'].list_collection_names()
    elif 'all-except-game-dates' in args.drop:
        drop = mongo_client['devel_ball'].list_collection_names()
        drop.remove('game-date')
        drop.remove('season')
    else:
        drop = args.drop
    for collection in drop:
        mongo_client['devel_ball'].drop_collection(collection)

    # Call the requested data acquiring functions
    if args.gamedates:
        get_gamedates(args.years)
    if args.games:
        get_games(args.years)
