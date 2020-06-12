#!/usr/bin/env python3.7

from argparse import ArgumentParser
from mongoengine import connect
import datetime
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd

from models import (
    PlayerSeason,
    TeamSeason,
    OfficialSeason,
    OfficialStatsPerGame,
)


def create_raw_dataframe(years):

    data = pd.DataFrame(columns=[
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
        'FG%', 'FG3%', 'FT%',

        # Player Advanced stats
        'AST_PCT', 'PER', 'USG_PCT', 'OFF_RTG', 'FLOOR_PCT', 'DEF_RTG', 'GAME_SCORE',
        'PIE', 'REB_PCT', 'OREB_PCT', 'DREB_PCT', 'AST_TOV', 'TO_PCT', 'eFG_PCT', 'TS_PCT',

        # Team traditional stats per game
        'PTStm', 'FGMtm', 'FGAtm', 'FG%tm', 'FG3Mtm', 'FG3Atm', 'FG3%tm', 'FTMtm', 'FTAtm', 'FT%TM',
        'OREBtm', 'DREBtm', 'REBtm', 'ASTtm', 'STLtm', 'BLKtm', 'TOtm', 'PFtm', 'PLUS_MINUStm',

        # Team advanced stats per game
        'AST_PCTtm', 'PACEtm', 'PIEtm', 'REB_PCTtm', 'OREB_PCTtm', 'DREB_PCTtm', 'AST_TOVtm',
        'TO_PCTtm', 'eFG_PCTtm', 'TS_PCTtm',

        # Opposing Team stats per game
        'PTSvsTm', 'FGMvsTm', 'FGAvsTm', 'FG%vsTm', 'FG3MvsTm', 'FG3AvsTm', 'FG3%vsTm',
        'FTMvsTm', 'FTAvsTm', 'FT%vsTm', 'OREBvsTm', 'DREBvsTm', 'REBvsTm', 'ASTvsTm',
        'STLvsTm', 'BLKvsTm', 'TOvsTm', 'PFvsTm', 'PLUS_MINUSvsTm',

        # Opposing Team advanced stats per game
        'AST_PCTvsTm', 'PACEvsTm', 'PIEvsTm', 'REB_PCTvsTm', 'OREB_PCTvsTm', 'DREB_PCTvsTm', 'AST_TOVvsTm',
        'TO_PCTvsTm', 'eFG_PCTvsTm', 'TS_PCTvsTm',

        # Official stats per game
        'PTSoff', 'FGAoff', 'FTAoff', 'POSSoff', 'PACEoff', 'eFG_PCToff', 'TS_PCToff',

        # Misc
        'HOME',

    ])

    player_seasons = PlayerSeason.objects(year__in=years)
    for player_season in player_seasons:
        season_stats = player_season.season_stats
        for game in season_stats:

            # If no player data for this game, skip
            if not game.per_game_stats or game.per_game_stats.MIN <= 0.0:
                continue

            # Get team data for this game, if none, skip
            team_season = TeamSeason.objects(year=player_season.year, team_id=game.team_id)[0]
            team_game = [team_game for team_game in team_season.season_stats
                         if team_game.game_id == game.game_id][0]
            if not team_game.traditional_stats_per_game:
                continue

            # Get opposing team data for this game, if none, skip
            vsTeam_season = TeamSeason.objects(year=player_season.year, team_id=game.opposing_team_id)[0]
            vsTeam_game = [team_game for team_game in team_season.season_stats
                           if team_game.game_id == game.game_id][0]
            if not vsTeam_game.traditional_stats_per_game:
                continue

            # Get official data for this game, if none, skip
            officials_season = OfficialSeason.objects(
                year=player_season.year, official_id__in=game.officials)
            official_games = [official_game.stats_per_game for official_season in officials_season
                              for official_game in official_season.season_stats
                              if official_game.game_id == game.game_id]
            if all(game is None for game in official_games):
                continue
            official_stats = OfficialStatsPerGame()
            for stat in official_stats:
                official_stats[stat] = np.mean([game[stat] for game in official_games if game])

            import IPython; IPython.embed()

            # Add POSS to team stats

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
