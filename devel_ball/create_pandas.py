from recordclass import recordclass
import datetime
from datetime import timedelta
from time import sleep
import numpy as np
import pandas as pd
import attr
import pickle

from draft_kings import client, Sport

from nba_api.stats.endpoints import CommonPlayerInfo

from .models import (
    Player,
    DraftKingsPlayer,
    Team,
    DraftKingsTeam,
    PlayerSeason,
    TeamSeason,
    OfficialSeason,
    OfficialStatsPerGame,
)
from .model_loader import query_nba_api


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
    #'PTSoff', 'FGAoff', 'FTAoff', 'POSSoff', 'PACEoff', 'eFG_PCToff', 'TS_PCToff',

    # Misc
    'HOME',
]


RECENT_VALUES = ['MIN', 'POSS', 'USG_PCT', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TO']


@attr.s
class DK_PLAYER(object):
    dk_player_entry = attr.ib()
    player_entry = attr.ib()
    team_entry = attr.ib()
    vs_team_entry = attr.ib()
    home = attr.ib()
    cost = attr.ib()
    positions = attr.ib()
    predicting = attr.ib(default=True)
    ineligible = attr.ib(default=False)
    ineligible_reason = attr.ib(default='')
    injury_status = attr.ib(default='')


def get_game_dict(player_id, date, player_stats, home, team_stats, vs_team_stats, results=None): #official_stats):

    game_dict = {value: None for value in GAME_VALUES}

    # Set basic accounting params
    game_dict['PLAYER_ID'] = player_id
    game_dict['DATE'] = date

    # Set results of the game if provided
    if results is not None:

        for key, value in results.to_mongo().to_dict().items():
            game_dict[key] = value

        # Update "calculated" result values
        game_dict["DK_POINTS_PER_MIN"] = game_dict["DK_POINTS"] / game_dict["MIN"] if (
            game_dict["MIN"] > 0.0
        ) else 0.0
        game_dict["DK_POINTS_PER_POSS"] = game_dict["DK_POINTS"] / game_dict["POSS"] if (
            game_dict["POSS"] > 0.0
        ) else 0.0

    # Set player traditional stats per game
    for key, value in player_stats.per_game.to_mongo().to_dict().items():
        game_dict[f'{key}pg'] = value
    game_dict['REBpg'] = game_dict['OREBpg'] + game_dict['DREBpg']
    game_dict['POSSpg'] = player_stats.advanced.to_mongo().to_dict()['POSS']

    # Set player traditional stats per minute
    for key, value in player_stats.per_minute.to_mongo().to_dict().items():
        game_dict[f'{key}pm'] = value
    game_dict['REBpm'] = game_dict['OREBpm'] + game_dict['DREBpm']

    # Set player traditional stats per possession
    for key, value in player_stats.per_possession.to_mongo().to_dict().items():
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
    for key, value in player_stats.advanced.to_mongo().to_dict().items():
        if key == 'POSS':
            continue
        game_dict[key] = value

    # Set team traditional stats
    for key, value in team_stats.per_game.to_mongo().to_dict().items():
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
    for key, value in team_stats.advanced.to_mongo().to_dict().items():
        game_dict[f'{key}tm'] = value

    # Set opposing team traditional stats
    for key, value in vs_team_stats.per_game.to_mongo().to_dict().items():
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
    for key, value in vs_team_stats.advanced.to_mongo().to_dict().items():
        game_dict[f'{key}vsTm'] = value

    # Set official stats
    #for key, value in official_stats.to_mongo().to_dict().items():
    #    game_dict[f'{key}off'] = value

    # Set home
    game_dict['HOME'] = home

    # Set recent values, 0 being most recent
    for recent_stat in RECENT_VALUES:
        for i, stat in enumerate(player_stats.recent['{}_RECENT_FIRST'.format(recent_stat)]):
            game_dict['RECENT_{}{}'.format(recent_stat, i)] = stat

    return game_dict


def create_training_dataframe(year, pickle_name):

    data = pd.DataFrame(columns=GAME_VALUES)

    player_seasons = PlayerSeason.objects(year=year)
    total_num = len(player_seasons)
    list_of_all_player_data = []
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
            #officials_season = OfficialSeason.objects(
            #    year=player_season.year, official_id__in=player_game.officials)
            #official_games = [official_game.stats_per_game for official_season in officials_season
            #                  for official_game in official_season.season_stats
            #                  if official_game.game_id == player_game.game_id]
            #if all(official_game is None for official_game in official_games):
            #    continue
            #official_stats = OfficialStatsPerGame()
            #for stat in official_stats:
            #    official_stats[stat] = np.mean(
            #        [official_game[stat] for official_game in official_games if official_game])

            game_dict = get_game_dict(
                player_id=player_season.player_id,
                date=player_game.date,
                player_stats=player_game.stats,
                home=player_game.home,
                team_stats=team_game.stats,
                vs_team_stats=vsTeam_game.stats,
                #official_stats,
                results=player_game.results,
            )
            list_of_all_player_data.append(pd.DataFrame([game_dict]))

    data = pd.concat([data, *list_of_all_player_data], ignore_index=True)
    data.to_pickle(pickle_name if pickle_name else f'{year}.p')


def get_player_from_name(first_name, last_name, year, previously_ignored=False, search_dk_players=False):
    """
    Helpful function to find the right player in the database built from nba_api from
    a string name from the draft_kings api
    """

    name = "{} {}".format(first_name, last_name)

    # Start with case insensitive exact search
    if search_dk_players:
        player = DraftKingsPlayer.objects(name__iexact=name)
    else:
        player = Player.objects(name__iexact=name, years__exists=year)
    if player and player.count() == 1:
        return player.first()

    # Next check if full name is contained inside of db name (case insensitive)
    if search_dk_players:
        player = DraftKingsPlayer.objects(name__icontains=name)
    else:
        player = Player.objects(name__icontains=name, years__exists=year)
    if player and player.count() == 1:
        return player.first()

    # If previously ignored, only do certain checks to not waste time with user input
    if previously_ignored:
        raise Exception("PLAYER NOT FOUND: {}".format(name))

    tested = []
    def check_match(player):
        match = player.id not in tested and str.upper(
            input(
                "Match?     Draftkings: {},   Database: {}, {}    (y/n)? ".format(name, player.name, player.id)
            )
        ) == 'Y'
        tested.append(player.id)
        return match

    # Next check if the first name or last name match 1 player in the db
    found = 0
    for part_name in [first_name, last_name]:
        if search_dk_players:
            player = DraftKingsPlayer.objects(name__icontains=part_name)
        else:
            player = Player.objects(name__icontains=part_name, years__exists=year)
        if player and player.count() == 1 and check_match(player.first()):
            return player.first()

    # Next check for each length of letters for last name and first name, if both are contained
    for i in range(len(first_name)):
        if search_dk_players:
            players = DraftKingsPlayer.objects(name__icontains=first_name[:i+1])
        else:
            players = Player.objects(name__icontains=first_name[:i+1], years__exists=year)
        for i in range(len(last_name)):
            quantity = 0
            found_player = None
            for player in players:
                if last_name[:i+1] in player.name:
                    quantity += 1
                    found_player = player
            if quantity == 1 and check_match(found_player):
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
    players_not_found = []
    for dk_player in client.available_players(dk_group_id)['players']:

        full_name = "{} {}".format(dk_player["first_name"], dk_player["last_name"])
        dk_player_entry = DraftKingsPlayer.objects(name=full_name).limit(1).first()

        # If this player has not been found before / stored as a dk_player
        if not dk_player_entry:
            try:
                player = get_player_from_name(dk_player["first_name"], dk_player["last_name"], year=year)
            except:
                players_not_found.append(full_name)
                continue
            dk_player_entry = DraftKingsPlayer(
                name=full_name,
                player=player,
            )
            dk_player_entry.save()
        # Else if this player has been deemed irrelevant in the past, search again, but if fail to find
        # gracefully move on
        elif dk_player_entry.player is None:
            try:
                player = get_player_from_name(
                    dk_player["first_name"], dk_player["last_name"], year=year, previously_ignored=True
                )
            except:
                pass
            else:
                print("Found previously ignored player, updating: {}".format(full_name))
                dk_player_entry.delete()
                dk_player_entry = DraftKingsPlayer(
                    name=full_name,
                    player=player,
                )
                dk_player_entry.save()

        # If player is relevant, make sure his team is in db
        if dk_player_entry.player is not None and not DraftKingsTeam.objects(
            dk_team_id=str(dk_player['team_id'])
        ).limit(1).first():
            common_player_info = query_nba_api(CommonPlayerInfo, player_id=dk_player_entry.player.unique_id)
            team_id = common_player_info.common_player_info.get_data_frame().TEAM_ID[0]
            team = Team.objects(unique_id=str(team_id)).limit(1).first()
            dk_team_entry = DraftKingsTeam(
                dk_team_id=str(dk_player['team_id']),
                team=team,
            )
            dk_team_entry.save()

        # Update player map for this player
        player_map[dk_player_entry] = dk_player

    # Wait until all possible players not found were determined, to raise this error once per day
    if players_not_found:
        still_not_found = []
        for player_not_found in players_not_found:
            if str.upper(input("Ignore?    {}    (y/n)? ".format(player_not_found))) == 'Y':
                # Create an ignorable player
                dk_player_entry = DraftKingsPlayer(name=player_not_found)
                dk_player_entry.save()
            else:
                still_not_found.append(player_not_found)
        if still_not_found:
            raise Exception("Players not found: \n\n{}".format("\n".join(still_not_found)))

    # Create final map to return
    final_players = []
    for dk_player_entry, dk_player in player_map.items():

        # If player is irrelevant, skip
        if dk_player_entry.player is None:
            player_entry = None
        else:
            player_entry = dk_player_entry.player

        # Get teams
        dk_team_id = str(dk_player['team_id'])
        home = dk_player['match_up']['home_team_id'] == dk_player['team_id']
        dk_vs_team_id = str(
            dk_player['match_up']['away_team_id'] if home else dk_player['match_up']['home_team_id']
        )
        dk_team = DraftKingsTeam.objects(dk_team_id=dk_team_id).limit(1).first()
        dk_vs_team = DraftKingsTeam.objects(dk_team_id=dk_vs_team_id).limit(1).first()

        # Update final map
        final_players.append(
            DK_PLAYER(
                dk_player_entry=dk_player_entry,
                player_entry=player_entry,
                team_entry=dk_team,
                vs_team_entry=dk_vs_team,
                home=home,
                cost=dk_player["draft"]["salary"],
                positions=[
                    position for position in ["PG", "SG", "SF", "PF", "C"]
                    if position in dk_player["position"]["name"]
                ],
            )
        )

    return final_players


def create_predicting_dataframe(year, pickle_name, today):

    # Get map of players to cost for given date
    date = datetime.datetime.today().date() if today else datetime.datetime.today().date() + timedelta(days=1)
    dk_players = get_draftkings_players_for_date(date, year)
    if dk_players is None:
        raise Exception("No draftkings contests found for {}".format(date))

    data = pd.DataFrame(columns=GAME_VALUES)
    for dk_player in dk_players:

        if dk_player.player_entry is None:
            print("{}: SKIPPING for no player".format(dk_player.dk_player_entry.name))
            dk_player.predicting = False
        else:
            player_season = PlayerSeason.objects(player_id=dk_player.player_entry.id, year=year).limit(1).first()
            team_season = TeamSeason.objects(team_id=dk_player.team_entry.team.id, year=year).limit(1).first()
            vs_team_season = TeamSeason.objects(team_id=dk_player.vs_team_entry.team.id, year=year).limit(1).first()
            if not player_season:
                print("{}: SKIPPING for no player_season".format(dk_player.dk_player_entry.name))
                dk_player.predicting = False
            if not team_season:
                print("{}: SKIPPING for no team_season".format(dk_player.dk_player_entry.name))
                dk_player.predicting = False
            if not vs_team_season:
                print("{}: SKIPPING for no vs_team_season".format(dk_player.dk_player_entry.name))
                dk_player.predicting = False

        if dk_player.predicting:
            game_dict = get_game_dict(
                player_id=dk_player.player_entry.id,
                date=date.isoformat(),
                player_stats=player_season.current_stats,
                home=dk_player.home,
                team_stats=team_season.current_stats,
                vs_team_stats=vs_team_season.current_stats,
            )
            data = data.append(game_dict, ignore_index=True)

    file_name = pickle_name if pickle_name else "predict.p"
    with open(file_name, 'wb') as f:
        pickle.dump((dk_players, data), f)
