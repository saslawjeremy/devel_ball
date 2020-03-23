#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
import datetime
from time import sleep

from nba_api.stats.static import players as nba_api_players
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    scoreboardv2
)

from models import (
    Player,
    GameDate,
    Season
)


def get_players(years):
    """
    Load the players info for the given years supplied

    :param years: the years to load info for
    :type  years: list[str]
    """

    # For each specified year, look at the players that played that year
    for year in years:

        # Pull a list of all player ids from this year
        year_stats = (
            leaguedashplayerstats.LeagueDashPlayerStats(season=year)
            .get_data_frames()[0]
        )
        players_info = [player_info for player_info in
                        zip(year_stats['PLAYER_ID'], year_stats['PLAYER_NAME'])
                       ]

        # If player already exists in database, update their years accordingly,
        # else, add player to database
        for player_id, player_name in players_info:

            player_entry = Player.objects(player_id=player_id)

            # Player already exists in database
            if player_entry:
                player_entry = player_entry[0]
                # If year isn't already stored for this player
                if year not in player_entry.years:
                    print("Adding {} to {} in the database"
                          .format(year, player_name))
                    player_entry.years.append(year)
                    player_entry.save()

            # Else player doesn't yet exist in database
            else:
                player_entry = Player()
                player_info = nba_api_players.find_player_by_id(player_id)
                player_entry.name = player_name
                player_entry.player_id = player_id
                player_entry.years = [year]
                player_entry.save()
                print("Adding {} to database".format(player_name))


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
        day_delta = datetime.timedelta(days=1)

        # Iterate until finding first day of regular season
        while True:
            sleep(1) # Don't overload querying
            gameday = scoreboardv2.ScoreboardV2(game_date=first_date)
            print("Looking at {} for first day of season".format(first_date))
            game_ids = gameday.available.get_data_frame()['GAME_ID']

            # If there were games this day, and it is regular season
            if len(game_ids)>0 and game_ids[0][2]=='2':
                season_entry.first_date = first_date
                break
            else:
                first_date = (
                    datetime.date.fromisoformat(first_date) + day_delta
                    ).isoformat()

        # Begin loading into mongo the game dates
        date = first_date
        while True:

            gamedate_entry = GameDate.objects(date=date)

            # Game date already exists in database
            if gamedate_entry:
                print('{} is already in the database'.format(date))
            # Else game date is not already in database
            else:
                sleep(1) # Don't overload querying
                gameday = scoreboardv2.ScoreboardV2(game_date=date)
                game_ids = (
                    gameday.available.get_data_frame()['GAME_ID'].to_list()
                )

                # If all star game, skip
                if len(game_ids)>0 and game_ids[0][2] == '3':
                    game_ids = []
                # If playoff game, stop and mark previous date as last day
                if len(game_ids)>0 and game_ids[0][2] == '4':
                    last_date = (
                        datetime.date.fromisoformat(date) - day_delta
                        ).isoformat()
                    season_entry.last_date = last_date
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

            date = (datetime.date.fromisoformat(date) + day_delta).isoformat()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to acquire player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--players', action='store_true',
                        help='Get player data to store under '
                             'Player collection.'
                       )
    parser.add_argument('--gamedates', action='store_true',
                        help='Get dates and games played on those dates.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    connect('devel_ball')

    # Call the requested data acquiring functions
    if args.players:
        get_players(args.years)
    if args.gamedates:
        get_gamedates(args.years)
