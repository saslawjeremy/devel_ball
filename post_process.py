#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
import datetime
from datetime import timedelta

from models import (
    Season,
    GameDate,
    Game,
)

def add_draftkings(years):
    """
    Add draftkings totals for each player for each game played in years

    :param years: the years to load draftkings info for
    :type  years: list[str]
    """

    for year in years:

        season = Season.objects(year=year)
        if not season:
            print('Season not yet loaded for {}'.format(year))
            continue
        season=season[0]
        first_date = season.first_date
        last_date = season.last_date

        # For each day in the season
        days = (datetime.date.fromisoformat(last_date) -
                datetime.date.fromisoformat(first_date)).days + 1
        for date in (datetime.date.fromisoformat(first_date) + timedelta(n)
                for n in range(days)):

            print('\n{}     Updating date: {}     {}\n'.format('#'*20, date, '#'*20))
            date=date.isoformat()

            # Fetch GameDate, if it doesn't exist then the season didn't
            # properly load and should re-load it
            game_date = GameDate.objects(date=date)
            if not game_date:
                print('GameDate not loaded for {}, you should re-load this '
                      'season {} in full to get the full season data before '
                      'proceeding.'.format(date, year))
                break
            game_date = game_date[0]

            # For each game on this day
            for game_id in game_date.games:

                print('game_id: {}'.format(game_id))

                # Fetch Game, if it exists already, skip it
                game = Game.objects(game_id=game_id)
                if not game:
                    print('Game {} not loaded for {}, stop and load this full '
                          'season data before adding draftkings data to it!'
                          .format(game_id, date))
                    break
                game = game[0]

                # Iterate over each player in that given game
                for player_id, player_game in game.player_games.items():
                    stats = player_game.traditional_stats
                    dk_points = (
                        1.0*stats.PTS
                        + 0.5*stats.FG3M
                        + 1.25*(stats.OREB + stats.DREB)
                        + 1.5*stats.AST
                        + 2.0*stats.STL
                        + 2.0*stats.BLK
                        - 0.5*stats.TO
                    )
                    double_digit_counter = 0
                    double_digit_counter += 1 if stats.PTS>=10.0 else 0
                    double_digit_counter += 1 if (stats.OREB + stats.DREB)>=10.0 else 0
                    double_digit_counter += 1 if stats.AST>=10.0 else 0
                    double_digit_counter += 1 if stats.BLK>=10.0 else 0
                    double_digit_counter += 1 if stats.STL>=10.0 else 0
                    if double_digit_counter == 2:
                        dk_points += 1.5
                    if double_digit_counter >= 3:
                        dk_points += 3.0

                    # Update player_game to reflect draftkings pointed scored
                    player_game.draftkings_points = dk_points

                # Save game to update draftkings points for each player in the game
                game.save()


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to post-process player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--draftkings', action='store_true',
                        help='Get team data to store under Team collection.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('devel_ball')

    # Call the requested data acquiring functions
    if args.draftkings:
        add_draftkings(args.years)
