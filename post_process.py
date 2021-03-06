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
    Official,
    OfficialStatsPerGame,
    OfficialSeasonDate,
    OfficialSeason,
    Team,
    TeamSeason,
    TeamSeasonDate,
    TeamAdvancedStatsPerGame,
    Player,
    PlayerSeason,
    PlayerSeasonDate,
    PlayerAdvancedStatsPerGame,
    PlayerResults,
    GameTraditionalStats
)

from stat_calculation_utils import *
from recordclass import recordclass

PlayerTotalSeasonStats = recordclass('PlayerTotalSeasonStats',
    ['GAMES', 'MIN', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
        'AST', 'STL', 'BLK', 'TO', 'PF', 'PLUS_MINUS', 'POSS',
     'TmMIN', 'TmPTS', 'TmFGM', 'TmFGA', 'TmFG3M', 'TmFG3A', 'TmFTM', 'TmFTA', 'TmOREB',
        'TmDREB', 'TmAST', 'TmSTL', 'TmBLK', 'TmTO', 'TmPF', 'TmPLUS_MINUS', 'TmPOSS',
     'vsTmMIN', 'vsTmPTS', 'vsTmFGM', 'vsTmFGA', 'vsTmFG3M', 'vsTmFG3A', 'vsTmFTM',
        'vsTmFTA', 'vsTmOREB', 'vsTmDREB', 'vsTmAST', 'vsTmSTL', 'vsTmBLK', 'vsTmTO',
        'vsTmPF', 'vsTmPLUS_MINUS', 'vsTmPOSS'],
    defaults=[0.0]*52
)

TeamTotalSeasonStats = recordclass('TeamTotalSeasonStats',
    ['GAMES', 'MIN', 'PTS', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB',
        'AST', 'STL', 'BLK', 'TO', 'PF', 'PLUS_MINUS', 'POSS',
     'vsMIN', 'vsPTS', 'vsFGM', 'vsFGA', 'vsFG3M', 'vsFG3A', 'vsFTM', 'vsFTA', 'vsOREB',
        'vsDREB', 'vsAST', 'vsSTL', 'vsBLK', 'vsTO', 'vsPF', 'vsPLUS_MINUS', 'vsPOSS'],
    defaults=[0.0]*35
)

TotalSeasonOfficialStats = recordclass('TotalSeasonOfficialStats',
    ['GAMES', 'MIN', 'PTS', 'FGM', 'FGA', 'FG3M', 'FTA', 'POSS'],
    defaults=[0.0]*8
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

def load_official_stats(stats_per_game, stats):
    stats_per_game.PTS = stats.PTS/stats.GAMES
    stats_per_game.FGA = stats.FGA/stats.GAMES
    stats_per_game.FTA = stats.FTA/stats.GAMES
    stats_per_game.POSS = stats.POSS/stats.GAMES
    stats_per_game.PACE = PACE(stats.MIN, stats.POSS/2.0, stats.POSS/2.0)
    stats_per_game.eFG_PCT = eFG_PCT(stats.FGM, stats.FG3M, stats.FGA)
    stats_per_game.TS_PCT = TS_PCT(stats.PTS, stats.FGA, stats.FTA)


def update_official_total_stats(total_stats, team_games):
    total_stats.GAMES += 1
    total_stats.MIN += team_games[0].traditional_stats.MIN
    total_stats.PTS += sum(game.traditional_stats.PTS for game in team_games)
    total_stats.FGM += sum(game.traditional_stats.FGM for game in team_games)
    total_stats.FGA += sum(game.traditional_stats.FGA for game in team_games)
    total_stats.FG3M += sum(game.traditional_stats.FG3M for game in team_games)
    total_stats.FTA += sum(game.traditional_stats.FTA for game in team_games)
    total_stats.POSS += sum(game.advanced_stats.POSS for game in team_games)


def add_official_season_data(years):
    """
    Create data for each official over the course of a given season

    :param years: the years to create team season data
    :type  years: list[str]
    """

    for year in years:

        # Get all officials that played in this year
        officials = Official.objects.filter(__raw__={f'years.{year}': {'$exists': True}})
        for official in officials:
            print(f"\n{'*'*20}     Loading {official.name} in year {year}     {'*'*20}\n")

            official_season = OfficialSeason()
            official_season.official_id = official.id
            official_season.year = year

            total_stats = TotalSeasonOfficialStats()

            # Iterate over each game that official was in this season
            for season_index, game_id in enumerate(official.years[year]):
                game = Game.objects(game_id=game_id)[0]

                season_date = OfficialSeasonDate()
                season_date.game_id = game_id
                season_date.date = game.date
                season_date.season_index = season_index

                if season_index > 0:
                    season_date.stats_per_game = OfficialStatsPerGame()
                    load_official_stats(season_date.stats_per_game, total_stats)
                else:
                    season_date.stats_per_game = None
                official_season.season_stats.append(season_date)

                # Get the stats from the game
                team_games = list(game.team_games.values())
                print(f"Game {season_index}: "
                      f"{Team.objects(unique_id=team_games[0].team_id)[0].name} vs. "
                      f"{Team.objects(unique_id=team_games[1].team_id)[0].name}")

                # Update total season stats for future calculations
                update_official_total_stats(total_stats, team_games)

            existing_official_season = OfficialSeason.objects(official_id=official.id, year=year)
            if len(existing_official_season) > 0:
                existing_official_season[0].delete()
            official_season.save()

def load_team_advanced_stats(game_advanced_stats, stats):
    game_advanced_stats.POSS = stats.POSS/stats.GAMES
    game_advanced_stats.AST_PCT = TmAST_PCT(stats.AST, stats.FGM)
    game_advanced_stats.PACE = PACE(stats.MIN, stats.POSS, stats.vsPOSS)
    game_advanced_stats.PIE = PIE(
        stats.PTS, stats.FGM, stats.FTM, stats.FGA, stats.FTA, stats.DREB, stats.OREB,
        stats.AST, stats.STL, stats.BLK, stats.PF, stats.TO, stats.PTS+stats.vsPTS,
        stats.FGM+stats.vsFGM, stats.FTM+stats.vsFTM, stats.FGA+stats.vsFGA,
        stats.FTA+stats.vsFTA, stats.DREB+stats.vsDREB, stats.OREB+stats.vsOREB,
        stats.AST+stats.vsAST, stats.STL+stats.vsSTL, stats.BLK+stats.vsBLK,
        stats.PF+stats.vsPF, stats.TO+stats.vsTO)
    game_advanced_stats.REB_PCT = REB_PCT(stats.MIN, stats.OREB+stats.DREB, stats.MIN*5.0,
        stats.OREB+stats.DREB, stats.vsOREB+stats.vsDREB)
    game_advanced_stats.OREB_PCT = REB_PCT(stats.MIN, stats.OREB, stats.MIN*5.0,
        stats.OREB, stats.vsDREB)
    game_advanced_stats.DREB_PCT = REB_PCT(stats.MIN, stats.DREB, stats.MIN*5.0,
        stats.DREB, stats.vsOREB)
    game_advanced_stats.AST_TOV = AST_TOV(stats.AST, stats.TO)
    game_advanced_stats.TO_PCT = TO_PCT(stats.TO, stats.FGA, stats.FTA)
    game_advanced_stats.eFG_PCT = eFG_PCT(stats.FGM, stats.FG3M, stats.FGA)
    game_advanced_stats.TS_PCT = TS_PCT(stats.PTS, stats.FGA, stats.FTA)


def update_team_total_stats(total_stats, team_game, vs_team_game):
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

            total_stats = TeamTotalSeasonStats()

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
                update_team_total_stats(total_stats, team_game, vs_team_game)

            existing_team_season = TeamSeason.objects(team_id=team.id, year=year)
            if len(existing_team_season) > 0:
                existing_team_season[0].delete()
            team_season.save()


def load_player_stats(season_date, stats):
    for stat in season_date.per_game_stats:
        season_date.per_game_stats[stat] = getattr(stats, stat)/stats.GAMES
    for stat in season_date.per_minute_stats:
        if stat != 'MIN':
            if stats.MIN > 0.0:
                season_date.per_minute_stats[stat] = getattr(stats, stat)/stats.MIN
            else:
                season_date.per_minute_stats[stat] = 0.0
    for stat in season_date.per_possession_stats:
        if stats.POSS > 0.0:
            season_date.per_possession_stats[stat] = getattr(stats, stat)/stats.POSS
        else:
            season_date.per_possession_stats[stat] = 0.0
    advanced_stats = season_date.advanced_stats_per_game
    advanced_stats.POSS = stats.POSS/stats.GAMES
    advanced_stats.AST_PCT = AST_PCT(stats.AST, stats.MIN, stats.FGM, stats.TmMIN, stats.TmFGM)
    advanced_stats.PER = PER(stats.FGM, stats.FGA, stats.STL, stats.FG3M, stats.FTM, stats.FTA,
        stats.BLK, stats.OREB, stats.AST, stats.DREB, stats.PF, stats.TO, stats.MIN)
    advanced_stats.USG_PCT = USG_PCT(stats.FGA, stats.FTA, stats.TO, stats.MIN, stats.TmMIN,
        stats.TmFGA, stats.TmFTA, stats.TmTO)
    advanced_stats.OFF_RTG = OFF_RTG(stats.MIN, stats.PTS, stats.FGM, stats.FGA, stats.FG3M,
        stats.FTM, stats.FTA, stats.OREB, stats.AST, stats.TO, stats.TmMIN, stats.TmPTS,
        stats.TmFGM, stats.TmFGA, stats.TmFG3M, stats.TmFTM, stats.TmFTA, stats.TmOREB,
        stats.TmAST, stats.TmTO, stats.vsTmDREB)
    advanced_stats.FLOOR_PCT = FLOOR_PCT(stats.MIN, stats.PTS, stats.FGM, stats.FGA,
        stats.FTM, stats.FTA, stats.OREB, stats.AST, stats.TO, stats.TmMIN, stats.TmPTS,
        stats.TmFGM, stats.TmFGA, stats.TmFTM, stats.TmFTA, stats.TmOREB, stats.TmAST,
        stats.TmTO, stats.vsTmDREB)
    advanced_stats.DEF_RTG = DEF_RTG(stats.MIN, stats.STL, stats.BLK, stats.DREB, stats.PF,
        stats.TmMIN, stats.TmDREB, stats.TmBLK, stats.TmSTL, stats.TmPF, stats.TmPOSS,
        stats.vsTmMIN, stats.vsTmPTS, stats.vsTmOREB, stats.vsTmFGM, stats.vsTmFGA,
        stats.vsTmFTM, stats.vsTmFTA, stats.vsTmTO)
    advanced_stats.GAME_SCORE = GAME_SCORE(stats.PTS, stats.FGM, stats.FGA, stats.FTM,
        stats.FTA, stats.OREB, stats.DREB, stats.STL, stats.AST, stats.BLK, stats.PF, stats.TO)
    advanced_stats.PIE = PIE(
        stats.PTS, stats.FGM, stats.FTM, stats.FGA, stats.FTA, stats.DREB, stats.OREB,
        stats.AST, stats.STL, stats.BLK, stats.PF, stats.TO, stats.TmPTS+stats.vsTmPTS,
        stats.TmFGM+stats.vsTmFGM, stats.TmFTM+stats.vsTmFTM, stats.TmFGA+stats.vsTmFGA,
        stats.TmFTA+stats.vsTmFTA, stats.TmDREB+stats.vsTmDREB, stats.TmOREB+stats.vsTmOREB,
        stats.TmAST+stats.vsTmAST, stats.TmSTL+stats.vsTmSTL, stats.TmBLK+stats.vsTmBLK,
        stats.TmPF+stats.vsTmPF, stats.TmTO+stats.vsTmTO)
    advanced_stats.REB_PCT = REB_PCT(stats.MIN, stats.OREB+stats.DREB, stats.TmMIN,
        stats.TmOREB+stats.TmDREB, stats.vsTmOREB+stats.vsTmDREB)
    advanced_stats.OREB_PCT = REB_PCT(stats.MIN, stats.OREB, stats.TmMIN, stats.TmOREB, stats.vsTmDREB)
    advanced_stats.DREB_PCT = REB_PCT(stats.MIN, stats.DREB, stats.TmMIN, stats.TmDREB, stats.vsTmOREB)
    advanced_stats.AST_TOV = AST_TOV(stats.AST, stats.TO)
    advanced_stats.TO_PCT = TO_PCT(stats.TO, stats.FGA, stats.FTA)
    advanced_stats.eFG_PCT = eFG_PCT(stats.FGM, stats.FG3M, stats.FGA)
    advanced_stats.TS_PCT = TS_PCT(stats.PTS, stats.FGA, stats.FTA)


def update_player_total_stats(total_stats, player_game, team_game, opposing_team_game):
    total_stats.GAMES += 1
    for stat in player_game.traditional_stats:
        total_stats[stat] += player_game.traditional_stats[stat]
    total_stats.POSS += player_game.advanced_stats.POSS
    for Tm_stat in team_game.traditional_stats:
        total_stats[f'Tm{Tm_stat}'] += team_game.traditional_stats[Tm_stat]
    total_stats.TmPOSS += team_game.advanced_stats.POSS
    for vsTm_stat in opposing_team_game.traditional_stats:
        total_stats[f'vsTm{vsTm_stat}'] += opposing_team_game.traditional_stats[vsTm_stat]
    total_stats.vsTmPOSS += opposing_team_game.advanced_stats.POSS


def add_player_season_data(years):
    """
    Create data for each player over the course of a given season

    :param years: the years to create player season data
    :type  years: list[str]
    """

    for year in years:

        # Get all players that played in this year
        players = Player.objects.filter(__raw__={f'years.{year}': {'$exists': True}})
        for player in players:
            print(f"\n{'*'*20}     Loading {player.name} in year {year}     {'*'*20}\n")

            player_season = PlayerSeason()
            player_season.player_id = player.id
            player_season.year = year

            total_stats = PlayerTotalSeasonStats()

            # Iterate over each game that player played in this season
            for season_index, game_id in enumerate(player.years[year]):
                game = Game.objects(game_id=game_id)[0]

                season_date = PlayerSeasonDate()
                season_date.game_id = game_id
                season_date.date = game.date
                season_date.season_index = season_index
                season_date.officials = game.officials

                if season_index > 0:
                    season_date.per_game_stats = GameTraditionalStats()
                    season_date.per_minute_stats = GameTraditionalStats()
                    season_date.per_possession_stats = GameTraditionalStats()
                    season_date.advanced_stats_per_game = PlayerAdvancedStatsPerGame()
                    load_player_stats(season_date, total_stats)
                else:
                    season_date.per_game_stats = None
                    season_date.per_minute_stats = None
                    season_date.per_possession_stats = None
                    season_date.advanced_stats_per_game = None

                # Get the stats of each team in the game
                player_game = game.player_games[player.id]
                season_date.home = player_game.home
                season_date.team_id = player_game.team_id
                season_date.opposing_team_id = player_game.opposing_team_id

                season_date.results = PlayerResults()
                season_date.results.DK_POINTS = player_game.draftkings_points
                season_date.results.MIN = player_game.traditional_stats.MIN
                season_date.results.POSS = player_game.advanced_stats.POSS

                print(f"Game {season_index}: "
                      f"{Team.objects(unique_id=player_game.team_id)[0].name} vs. "
                      f"{Team.objects(unique_id=player_game.opposing_team_id)[0].name}")

                player_season.season_stats.append(season_date)

                # Update total season stats for future calculations
                team_game = game.team_games[player_game.team_id]
                opposing_team_game = game.team_games[player_game.opposing_team_id]
                update_player_total_stats(total_stats, player_game, team_game, opposing_team_game)

            existing_player_season = PlayerSeason.objects(player_id=player.id, year=year)
            if len(existing_player_season) > 0:
                existing_player_season[0].delete()
            player_season.save()

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to post-process player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    parser.add_argument('--draftkings', action='store_true',
                        help='Update each game to have draftkings points for each player.'
                       )
    parser.add_argument('--player-seasons', action='store_true',
                        help='Create day by day team statistics for each player in the given years.'
                       )
    parser.add_argument('--team-seasons', action='store_true',
                        help='Create day by day team statistics for each team in the given years.'
                       )
    parser.add_argument('--official-seasons', action='store_true',
                        help='Create day by day team statistics for each official in the given years.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('devel_ball')

    # Call the requested data acquiring functions
    if args.draftkings:
        add_draftkings(args.years)
    if args.official_seasons:
        add_official_season_data(args.years)
    if args.team_seasons:
        add_team_season_data(args.years)
    if args.player_seasons:
        add_player_season_data(args.years)
