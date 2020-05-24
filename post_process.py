#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
import datetime
from datetime import timedelta
from collections import namedtuple

from models import (
    Season,
    GameDate,
    Game,
    Team,
    TeamSeason,
    TeamSeasonDate,
    TeamAdvancedStatsPerGame,
    GameTraditionalStats
)

from stat_calculation_utils import *
from recordclass import recordclass

TotalSeasonStats = recordclass('TotalSeasonStats',
    ['GAMES', 'MIN', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
        'AST', 'STL', 'BLK', 'TO', 'PF', 'PLUS_MINUS', 'POSS', 'vsMIN', 'vsPTS', 'vsFGM',
        'vsFGA', 'vsFG3M', 'vsFG3A', 'vsFTM', 'vsFTA', 'vsOREB', 'vsDREB', 'vsAST',
        'vsSTL', 'vsBLK', 'vsTO', 'vsPF', 'vsPLUS_MINUS', 'vsPOSS'],
    defaults=[0.0]*35
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

def load_team_advanced_stats(game_advanced_stats, stats):
    game_advanced_stats.AST_PCT = TmAST_PCT(stats.AST, stats.FGM)
    game_advanced_stats.PACE = PACE(stats.MIN, stats.POSS, stats.vsPOSS)
    import IPython; IPython.embed()
    game_advanced_stats.PIE = PIE(
        stats.PTS, stats.FGM, stats.FTM, stats.FTA, stats.DREB, stats.OREB, stats.AST, stats.STL,
        stats.BLK, stats.PF, stats.TO, stats.PTS+stats.vsPTS, stats.FGM+stats.vsFGM,
        stats.FTM+stats.vsFTM, stats.FGA+stats.vsFGA, stats.FTA+stats.vsFTA, stats.DREB+stats.vsDREB,
        stats.OREB+stats.vsOREB, stats.AST+stats.vsAST, stats.STL+stats.vsSTL, stats.BLK+stats.vsBLK,
        stats.PF+stats.vsPF, stats.TO+stats.vsTO)


def update_total_stats(total_stats, team_game, vs_team_game):
    total_stats.GAMES += 1
    for stat in team_game.traditional_stats:
        total_stats[stat] += team_game.traditional_stats[stat]
    total_stats.POSS += team_game.advanced_stats.POSS
    for vs_stat in vs_team_game.traditional_stats:
        total_stats[f'vs{vs_stat}'] += vs_team_game.traditional_stats[vs_stat]
    total_stats.vsPOSS += vs_team_game.advanced_stats.POSS


def add_team_season_data(years):
    """
    Create data for each team over the course of a given season

    :param years: the years to create team season data
    :type  years: list[str]
    """

    for year in years:

        # Get all teams that played in this year
        teams = Team.objects.filter(__raw__={f'years.{year}': {'$exists': True}})
        for team in teams:
            print(f"\n{'*'*20}     Loading {team.name} in year {year}     {'*'*20}\n")

            team_season = TeamSeason()
            team_season.team_id = team.id
            team_season.year = year

            total_stats = TotalSeasonStats()

            # Iterate over each game that team played in this season
            for season_index, game_id in enumerate(team.years[year]):
                game = Game.objects(game_id=game_id)[0]

                season_date = TeamSeasonDate()
                season_date.game_id = game_id
                season_date.date = game.date
                season_date.season_index = season_index
                season_date.officials = game.officials

                if season_index > 0:
                    season_date.traditional_stats_per_game = GameTraditionalStats()
                    for stat in season_date.traditional_stats_per_game:
                        season_date.traditional_stats_per_game[stat] = (
                            getattr(total_stats, stat)/total_stats.GAMES)

                    season_date.advanced_stats_per_game = TeamAdvancedStatsPerGame()
                    load_team_advanced_stats(season_date.advanced_stats_per_game, total_stats)
                else:
                    season_date.traditional_stats_per_game = None
                    season_date.advanced_stats_per_game = None

                # Get the stats of each team in the game
                team_game = game.team_games[team.id]
                season_date.home = team_game.home
                season_date.opposing_team_id = team_game.opposing_team_id
                vs_team_game = game.team_games[team_game.opposing_team_id]
                print(f"Game {season_index}: {team.name} vs. "
                      f"{Team.objects(unique_id=team_game.opposing_team_id)[0].name}")

                team_season.season_stats.append(season_date)

                # Update total season stats for future calculations
                update_total_stats(total_stats, team_game, vs_team_game)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to post-process player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--draftkings', action='store_true',
                        help='Update each game to have draftkings points for each player.'
                       )
    parser.add_argument('--team-seasons', action='store_true',
                        help='Create day by day team statistics for each team in the given years.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('devel_ball')

    # Call the requested data acquiring functions
    if args.draftkings:
        add_draftkings(args.years)
    if args.team_seasons:
        add_team_season_data(args.years)
