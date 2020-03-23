#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect

from nba_api.stats.static import players as nba_api_players
from nba_api.stats.endpoints import leaguedashplayerstats as player_stats

from models import Player


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
            player_stats.LeagueDashPlayerStats(season=year)
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
                print("Adding {} to database".format(player_name))
                player_entry = Player()
                player_info = nba_api_players.find_player_by_id(player_id)
                player_entry.name = player_name
                player_entry.player_id = player_id
                player_entry.years = [year]
                player_entry.save()


def get_dates(years):
    """
    Load the dates and games played on them for the given years supplied

    :param years: the years to load info for
    :type  years: list[str]
    """

    # For each specified year, look at the dates and games played on them
    for year in years:

        # Pull a list of all player ids from this year
        year_stats = (
            player_stats.LeagueDashPlayerStats(season=year)
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
                print("Adding {} to database".format(player_name))
                player_entry = Player()
                player_info = nba_api_players.find_player_by_id(player_id)
                player_entry.name = player_name
                player_entry.player_id = player_id
                player_entry.years = [year]
                player_entry.save()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2019-20'], nargs='*',
                        help='Years to acquire player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--players', action='store_true',
                        help='Get player data to store under '
                             'Player collection.'
                       )
    parser.add_argument('--dates', action='store_true',
                        help='Get dates and games played on those dates.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    connect('devel_ball')

    # Call the requested data acquiring functions
    if args.players:
        get_players(args.years)
    if args.dates:
        get_dates(args.years)
