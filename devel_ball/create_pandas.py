from recordclass import recordclass
import datetime
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd

from draft_kings import client, Sport

from .models import (
    Player,
    DraftKingsPlayer,
    PlayerSeason,
    TeamSeason,
    OfficialSeason,
    OfficialStatsPerGame,
)


GAME_VALUES = [

    # Basic accounting
    'PLAYER_ID', 'DATE',

    # Things to predict
    'DK_POINTS', 'MIN', 'POSS', 'DK_POINTS_PER_MIN', 'DK_POINTS_PER_POSS',

    # Player traditional stats per game
    'MINpg', 'POSSpg', 'PTSpg', 'FGMpg', 'FGApg', 'FG3Mpg', 'FG3Apg', 'FTMpg', 'FTApg',
    'OREBpg', 'DREBpg', 'REBpg', 'ASTpg', 'STLpg', 'BLKpg', 'TOpg', 'PFpg', 'PLUS_MINUSpg',

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

RECENT_VALUES = ['MIN', 'POSS', 'USG_PCT', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO']


def get_game_dict(player_id, player_game, team_game, vsTeam_game, official_stats):

    game_dict = {value: None for value in GAME_VALUES}

    # Set basic accounting params
    game_dict['PLAYER_ID'] = player_id
    game_dict['DATE'] = player_game.date

    # Set results of the game
    for key, value in player_game.results.to_mongo().iteritems():
        game_dict[key] = value

    # Set player traditional stats per game
    for key, value in player_game.stats.per_game.to_mongo().iteritems():
        game_dict[f'{key}pg'] = value
    game_dict['REBpg'] = game_dict['OREBpg'] + game_dict['DREBpg']
    game_dict['POSSpg'] = player_game.stats.advanced.to_mongo()['POSS']

    # Set player traditional stats per minute
    for key, value in player_game.stats.per_minute.to_mongo().iteritems():
        game_dict[f'{key}pm'] = value
    game_dict['REBpm'] = game_dict['OREBpm'] + game_dict['DREBpm']

    # Set player traditional stats per possession
    for key, value in player_game.stats.per_possession.to_mongo().iteritems():
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
    for key, value in player_game.stats.advanced.to_mongo().iteritems():
        if key == 'POSS':
            continue
        game_dict[key] = value

    # Set team traditional stats
    for key, value in team_game.stats.per_game.to_mongo().iteritems():
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
    for key, value in team_game.stats.advanced.to_mongo().iteritems():
        game_dict[f'{key}tm'] = value

    # Set opposing team traditional stats
    for key, value in vsTeam_game.stats.per_game.to_mongo().iteritems():
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
    for key, value in vsTeam_game.stats.advanced.to_mongo().iteritems():
        game_dict[f'{key}vsTm'] = value

    # Set official stats
    for key, value in official_stats.to_mongo().iteritems():
        game_dict[f'{key}off'] = value

    # Set home
    game_dict['HOME'] = player_game.home

    # Set recent values, 0 being most recent
    for recent_stat in RECENT_VALUES:
        for i, stat in enumerate(player_game.stats.recent['{}_RECENT_FIRST'.format(recent_stat)]):
            game_dict['RECENT_{}{}'.format(recent_stat, i)] = stat

    # Update "calculated" result values
    game_dict["DK_POINTS_PER_MIN"] = game_dict["DK_POINTS"] / game_dict["MIN"] if (
        game_dict["MIN"] > 0.0
    ) else 0.0
    game_dict["DK_POINTS_PER_POSS"] = game_dict["DK_POINTS"] / game_dict["POSS"] if (
        game_dict["POSS"] > 0.0
    ) else 0.0

    return game_dict


def create_training_dataframe(years, pickle_name):

    data = pd.DataFrame(columns=GAME_VALUES)

    player_seasons = PlayerSeason.objects(year__in=years)
    total_num = len(player_seasons)
    for i, player_season in enumerate(player_seasons):
        print(f'Loading {Player.objects(pk=player_season.player_id)[0].name} in '
              f'{player_season.year}   ({i}/{total_num})')
        season_stats = player_season.season_stats

        for player_game in season_stats:

            # If no player data for this game, skip
            if not player_game.stats or player_game.stats.per_game.MIN <= 0.0:
                continue

            # Get team data for this game, if none, skip
            team_season = TeamSeason.objects(year=player_season.year,
                                             team_id=player_game.team_id)[0]
            team_game = [team_game for team_game in team_season.season_stats
                         if team_game.game_id == player_game.game_id][0]
            if not team_game.stats:
                continue

            # Get opposing team data for this game, if none, skip
            vsTeam_season = TeamSeason.objects(year=player_season.year,
                                               team_id=player_game.opposing_team_id)[0]
            vsTeam_game = [vsTeam_game for vsTeam_game in vsTeam_season.season_stats
                           if vsTeam_game.game_id == player_game.game_id][0]
            if not vsTeam_game.stats:
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

            game_dict = get_game_dict(
                player_season.player_id, player_game, team_game, vsTeam_game, official_stats
            )
            data = data.append(game_dict, ignore_index=True)

    data.to_pickle(pickle_name if pickle_name else f'{years}.p')


def get_player_from_name(first_name, last_name, year):
    """
    Helpful function to find the right player in the database built from nba_api from
    a string name from the draft_kings api
    """

    name = "{} {}".format(first_name, last_name)

    # Start with case insensitive exact search
    player = Player.objects(name__iexact=name, years__exists=year)
    if player and player.count() == 1:
        return player.first()

    # Next check if full name is contained inside of db name (case insensitive)
    player = Player.objects(name__icontains=name, years__exists=year)
    if player and player.count() == 1:
        return player.first()

    # Next check if any part of the name is found inside just 1 player in db
    for part_name in [first_name, last_name]:
        player = Player.objects(name__icontains=part_name, years__exists=year)
        if player and player.count() == 1:
            return player.first()

    # Next check for each length of letters for last name and first name, if both are contained
    for i in range(len(first_name)):
        players = Player.objects(name__icontains=first_name[:i+1], years__exists=year)
        for i in range(len(last_name)):
            quantity = 0
            found_player = None
            for player in players:
                if last_name[:i+1] in player.name:
                    quantity += 1
                    found_player = player
            if quantity == 1:
                return found_player

    raise Exception("PLAYER NOT FOUND: {}".format(name))


def get_draftkings_players_for_date(date, year):

    # Find the contest with the players of relevance for date, by searching
    # for the player group with the most games within it for that given day
    games_count = 0
    dk_group_id = None
    for group in client.contests(sport=Sport.NBA)['groups']:

        # Skip groups from the wrong data
        group_date = group['starts_at'].astimezone().date()
        if group_date != date:
            continue

        # Find the group with the most games to play in this day
        if group["games_count"] > games_count:
            dk_group_id = group["id"]
            games_count = group["games_count"]

    if dk_group_id is None:
        print("No group found")
        return

    # Find the players for this group
    player_map = {}
    for dk_player in client.available_players(dk_group_id)['players']:

        full_name = "{} {}".format(dk_player["first_name"], dk_player["last_name"])
        dk_player_entry = DraftKingsPlayer.objects(dk_name=full_name).limit(1).first()
        # If this player has not been found before / stored as a dk_player
        if not dk_player_entry:
            player = get_player_from_name(dk_player["first_name"], dk_player["last_name"], year=year)
            dk_player_entry = DraftKingsPlayer(
                dk_name=full_name,
                player=player,
            )
            dk_player_entry.save()
        # Else if this player has been deemed irrelevant in the past, search again, but if fail to find
        # gracefully move on
        elif dk_player_entry.player is None:
            try:
                player = get_player_from_name(dk_player["first_name"], dk_player["last_name"], year=year)
                print("Found previously ignored player, updating: {}".format(full_name))
            except:
                pass
            else:
                dk_player_entry.delete()
                dk_player_entry = DraftKingsPlayer(
                    dk_name=full_name,
                    player=player,
                )
                dk_player_entry.save()

        player_map[dk_player_entry] = dk_player["draft"]["salary"]

    return player_map

def create_predicting_dataframe(years, pickle_name, today=False):

    # Get map of players to cost for given date
    date = datetime.datetime.today().date() if today else datetime.datetime.today().date() + timedelta(days=1)
    dk_players = get_draftkings_players_for_date(date, years[0])



    ### DEBUG ###
    for dk_player, cost in dk_players.items():
        print(dk_player.dk_name, dk_player.player.name if dk_player.player is not None else "NONE", cost)
    ###
