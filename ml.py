#!/usr/bin/env python3.7

from recordclass import recordclass
from argparse import ArgumentParser
from mongoengine import connect
import datetime
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd

from models import (
    Player,
    PlayerSeason,
    TeamSeason,
    OfficialSeason,
    OfficialStatsPerGame,
)


GAME_VALUES = [
    # Things to predict
    'DK_POINTS', 'MIN', 'POSS',

    # Player traditional stats per game
    'PTSpg', 'FGMpg', 'FGApg', 'FG3Mpg', 'FG3Apg', 'FTMpg', 'FTApg', 'OREBpg',
    'DREBpg', 'REBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'PLUS_MINUSpg',

    # Player traditional stats per minute
    'PTSpm', 'FGMpm', 'FGApm', 'FG3Mpm', 'FG3Apm', 'FTMpm', 'FTApm', 'OREBpm',
    'DREBpm', 'REBpm', 'ASTpm', 'STLpm', 'BLKpm', 'TOpm', 'PFpm', 'PLUS_MINUSpm',

    # Player traditional stats per possession
    'PTSpp', 'FGMpp', 'FGApp', 'FG3Mpp', 'FG3App', 'FTMpp', 'FTApp', 'OREBpp',
    'DREBpp', 'REBpp', 'ASTpp', 'STLpp', 'BLKpp', 'TOpp', 'PFpp', 'PLUS_MINUSpp',

    # Player traditional stats (rates)
    'FGP', 'FG3P', 'FTP',

    # Player Advanced stats
    'AST_PCT', 'PER', 'USG_PCT', 'OFF_RTG', 'FLOOR_PCT', 'DEF_RTG', 'GAME_SCORE',
    'PIE', 'REB_PCT', 'OREB_PCT', 'DREB_PCT', 'AST_TOV', 'TO_PCT', 'eFG_PCT', 'TS_PCT',

    # Team traditional stats per game
    'PTStm', 'FGMtm', 'FGAtm', 'FGPtm', 'FG3Mtm', 'FG3Atm', 'FG3Ptm', 'FTMtm', 'FTAtm', 'FTPtm',
    'OREBtm', 'DREBtm', 'REBtm', 'ASTtm', 'STLtm', 'BLKtm', 'TOtm', 'PFtm', 'PLUS_MINUStm',

    # Team advanced stats per game
    'POSStm', 'AST_PCTtm', 'PACEtm', 'PIEtm', 'REB_PCTtm', 'OREB_PCTtm', 'DREB_PCTtm',
    'AST_TOVtm', 'TO_PCTtm', 'eFG_PCTtm', 'TS_PCTtm',

    # Opposing Team stats per game
    'PTSvsTm', 'FGMvsTm', 'FGAvsTm', 'FGPvsTm', 'FG3MvsTm', 'FG3AvsTm', 'FG3PvsTm',
    'FTMvsTm', 'FTAvsTm', 'FTPvsTm', 'OREBvsTm', 'DREBvsTm', 'REBvsTm', 'ASTvsTm',
    'STLvsTm', 'BLKvsTm', 'TOvsTm', 'PFvsTm', 'PLUS_MINUSvsTm',

    # Opposing Team advanced stats per game
    'POSSvsTm', 'AST_PCTvsTm', 'PACEvsTm', 'PIEvsTm', 'REB_PCTvsTm', 'OREB_PCTvsTm',
    'DREB_PCTvsTm', 'AST_TOVvsTm', 'TO_PCTvsTm', 'eFG_PCTvsTm', 'TS_PCTvsTm',

    # Official stats per game
    'PTSoff', 'FGAoff', 'FTAoff', 'POSSoff', 'PACEoff', 'eFG_PCToff', 'TS_PCToff',

    # Misc
    'HOME',
]


def get_game_dict(player_game, team_game, vsTeam_game, official_stats):

    game_dict = {value: None for value in GAME_VALUES}

    # Set results of the game
    for key, value in player_game.results.to_mongo().iteritems():
        game_dict[key] = value

    # Set player traditional stats per game
    for key, value in player_game.per_game_stats.to_mongo().iteritems():
        if key == 'MIN':
            continue
        game_dict[f'{key}pg'] = value
    game_dict['REBpg'] = game_dict['OREBpg'] + game_dict['DREBpg']

    # Set player traditional stats per minute
    for key, value in player_game.per_minute_stats.to_mongo().iteritems():
        game_dict[f'{key}pm'] = value
    game_dict['REBpm'] = game_dict['OREBpm'] + game_dict['DREBpm']

    # Set player traditional stats per possession
    for key, value in player_game.per_possession_stats.to_mongo().iteritems():
        if key == 'MIN':
            continue
        game_dict[f'{key}pp'] = value
    game_dict['REBpp'] = game_dict['OREBpp'] + game_dict['DREBpp']

    # Set player traditional stats (rates)
    game_dict['FGP'] = (game_dict['FGMpg'] / game_dict['FGApg']
                        if game_dict['FGApg'] > 0.0 else 0.0)
    game_dict['FG3P'] = (game_dict['FG3Mpg'] / game_dict['FG3Apg']
                         if game_dict['FG3Apg'] > 0.0 else 0.0)
    game_dict['FTP'] = (game_dict['FTMpg'] / game_dict['FTApg']
                        if game_dict['FTApg'] > 0.0 else 0.0)

    # Set player advanced stats
    for key, value in player_game.advanced_stats_per_game.to_mongo().iteritems():
        game_dict[key] = value

    # Set team traditional stats
    for key, value in team_game.traditional_stats_per_game.to_mongo().iteritems():
        if key == 'MIN':
            continue
        game_dict[f'{key}tm'] = value
    game_dict['REBtm'] = game_dict['OREBtm'] + game_dict['DREBtm']
    game_dict['FGPtm'] = (game_dict['FGMtm'] / game_dict['FGAtm']
                          if game_dict['FGAtm'] > 0.0 else 0.0)
    game_dict['FG3Ptm'] = (game_dict['FG3Mtm'] / game_dict['FG3Atm']
                           if game_dict['FG3Atm'] > 0.0 else 0.0)
    game_dict['FTPtm'] = (game_dict['FTMtm'] / game_dict['FTAtm']
                          if game_dict['FTAtm'] > 0.0 else 0.0)

    # Set team advanced stats
    for key, value in team_game.advanced_stats_per_game.to_mongo().iteritems():
        game_dict[f'{key}tm'] = value

    # Set opposing team traditional stats
    for key, value in vsTeam_game.traditional_stats_per_game.to_mongo().iteritems():
        if key == 'MIN':
            continue
        game_dict[f'{key}vsTm'] = value
    game_dict['REBvsTm'] = game_dict['OREBvsTm'] + game_dict['DREBvsTm']
    game_dict['FGPvsTm'] = (game_dict['FGMvsTm'] / game_dict['FGAvsTm']
                            if game_dict['FGAvsTm'] > 0.0 else 0.0)
    game_dict['FG3PvsTm'] = (game_dict['FG3MvsTm'] / game_dict['FG3AvsTm']
                             if game_dict['FG3AvsTm'] > 0.0 else 0.0)
    game_dict['FTPvsTm'] = (game_dict['FTMvsTm'] / game_dict['FTAvsTm']
                            if game_dict['FTAvsTm'] > 0.0 else 0.0)

    # Set opposing team advanced stats
    for key, value in vsTeam_game.advanced_stats_per_game.to_mongo().iteritems():
        game_dict[f'{key}vsTm'] = value

    # Set official stats
    for key, value in official_stats.to_mongo().iteritems():
        game_dict[f'{key}off'] = value

    # Set home
    game_dict['HOME'] = player_game.home

    return game_dict


def create_raw_dataframe(years):

    data = pd.DataFrame(columns=GAME_VALUES)

    player_seasons = PlayerSeason.objects(year__in=years)
    for player_season in player_seasons:

        print(f'Loading {Player.objects(pk=player_season.player_id)[0].name} in {player_season.year}')
        season_stats = player_season.season_stats

        for player_game in season_stats:

            # If no player data for this game, skip
            if not player_game.per_game_stats or player_game.per_game_stats.MIN <= 0.0:
                continue

            # Get team data for this game, if none, skip
            team_season = TeamSeason.objects(year=player_season.year,
                                             team_id=player_game.team_id)[0]
            team_game = [team_game for team_game in team_season.season_stats
                         if team_game.game_id == player_game.game_id][0]
            if not team_game.traditional_stats_per_game:
                continue

            # Get opposing team data for this game, if none, skip
            vsTeam_season = TeamSeason.objects(year=player_season.year,
                                               team_id=player_game.opposing_team_id)[0]
            vsTeam_game = [team_game for team_game in team_season.season_stats
                           if team_game.game_id == player_game.game_id][0]
            if not vsTeam_game.traditional_stats_per_game:
                continue

            # Get official data for this game, if none, skip
            officials_season = OfficialSeason.objects(
                year=player_season.year, official_id__in=player_game.officials)
            official_games = [official_game.stats_per_game for official_season in officials_season
                              for official_game in official_season.season_stats
                              if official_game.game_id == player_game.game_id]
            if all(official_game is None for official_game in official_games):
                continue
            official_stats = OfficialStatsPerGame()
            for stat in official_stats:
                official_stats[stat] = np.mean(
                    [official_game[stat] for official_game in official_games if official_game])

            game_dict = get_game_dict(player_game, team_game, vsTeam_game, official_stats)
            data = data.append(game_dict, ignore_index=True)

        print(data)

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--years', default=['2018-19'], nargs='*',
                        help='Years to acquire player data for. Please use the'
                             ' format xxxx-xx, e.g. 2019-20.'
                       )
    args = parser.parse_args()

    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('devel_ball')
    create_raw_dataframe(args.years)
