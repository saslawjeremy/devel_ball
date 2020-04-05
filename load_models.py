#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
from datetime import (
    timedelta
)
import datetime
from time import sleep

from nba_api.stats.static import (
    players as nba_api_players,
    teams as nba_api_teams
)
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


def query_nba_api(endpoint, sleep_time=1, **kwargs):
    """
    Query the nba_api at a safe rate

    :param endpoint: the nba_api endpoint to utilize
    :type  endpoint: function
    :param sleep_time: amount to sleep before lookup to not overload the server
    :type  sleep_time: int
    """
    print("Querying nba_api {} with args: {}".format(
        str(endpoint).split(".")[-1].split('\'')[0],
        kwargs)
    )
    sleep(sleep_time)
    return endpoint(**kwargs)


def add_player_to_db(player_id, player_name, year):
    """
    Add player to db for given year if not already stored

    :param player_id: player's unique id
    :type  player_id: str
    :param player_name: player's full name
    :type  player_name: str
    :param year: year that player is playing
    :type  year: str
    """

    player_entry = Player.objects(player_id=player_id)
    if player_entry:
        player_entry = player_entry[0]
        if year not in player_entry.years:
            player_entry.years.append(year)
            player_entry.save()
    else:
        player_entry = Player(player_id=player_id)
        player_entry.name = player_name
        player_entry.years = [year]
        player_entry.save()


def get_teams():
    """
    Load the teams into the database
    """

    teams = nba_api_teams.get_teams()
    for team in teams:
        team_entry = Team.objects(team_id=team['id'])
        if team_entry:
            print("{} already in database.".format(team['full_name']))
        else:
            team_entry = Team()
            team_entry.team_id = str(team['id'])
            team_entry.name = team['full_name']
            team_entry.save()
            print("Adding {} to database.".format(team_entry.name))


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
            for game_id in game_date.games:

                # Fetch Game, if it exists already, skip it
                game = Game.objects(game_id=game_id)
                if game:
                    continue
                game = Game(game_id=game_id)

                # Fetch Box Score Summary
                box_score_summary = query_nba_api(
                    boxscoresummaryv2.BoxScoreSummaryV2, game_id=game_id)

                # Store inactive players
                game.inactives = [
                    str(inactive_player) for inactive_player in
                    box_score_summary.inactive_players.get_data_frame()
                        ['PLAYER_ID'].to_list()
                ]

                # Store officials for this game (create Official if needed)
                officials = []
                for _, official in box_score_summary.officials.get_data_frame().iterrows():
                    official_id = str(official['OFFICIAL_ID'])
                    official_entry = Official.objects(official_id=official_id)
                    if not official_entry:
                        official_entry = Official(official_id=official_id)
                        official_entry.name = '{} {}'.format(official['FIRST_NAME'],
                                                             official['LAST_NAME'])
                        official_entry.save()
                    officials.append(official_id)
                game.officials = officials

                # Store home team id and road team id
                game_summary = box_score_summary.game_summary.get_data_frame()
                home_team_id = str(game_summary['HOME_TEAM_ID'][0])
                road_team_id = str(game_summary['VISITOR_TEAM_ID'][0])

                # Fetch various relevant box scores to use
                box_score_traditional = query_nba_api(
                    boxscoretraditionalv2.BoxScoreTraditionalV2, game_id=game_id)
                players_traditional = box_score_traditional.player_stats.get_data_frame()
                teams_traditional = box_score_traditional.team_stats.get_data_frame()
                box_score_advanced = query_nba_api(
                    boxscoreadvancedv2.BoxScoreAdvancedV2, game_id=game_id)
                players_advanced = box_score_advanced.player_stats.get_data_frame()
                teams_advanced = box_score_advanced.team_stats.get_data_frame()
                box_score_usage = query_nba_api(
                    boxscoreusagev2.BoxScoreUsageV2, game_id=game_id)
                players_usage = box_score_usage.sql_players_usage.get_data_frame()

                print('\n{}     Loading game: {} vs. {}     {}'
                      .format('#'*10, Team.objects(team_id=home_team_id)[0].name,
                              Team.objects(team_id=road_team_id)[0].name, '#'*10))

                # Create each PlayerGame and map them to player_id
                player_games = {}
                for _, player in players_traditional.iterrows():

                    # Gather player info and add to db for this year if not already stored
                    player_id = str(player['PLAYER_ID'])
                    player_name = str(player['PLAYER_NAME'])
                    print("Player: {}  (id: {})".format(player_name, player_id))
                    add_player_to_db(player_id, player_name, year)


                    player_game = PlayerGame(player_id=player_id)

                print("")


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to acquire player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--teams', action='store_true',
                        help='Get team data to store under Team collection.'
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
    connect('devel_ball')

    # Call the requested data acquiring functions
    if args.teams:
        get_teams()
    if args.gamedates:
        get_gamedates(args.years)
    if args.games:
        get_games(args.years)
