import pandas as pd
import bisect
import attr
import numpy as np
from copy import deepcopy

from devel_ball.models import (
    Game,
    TeamSeason,
    PlayerSeason,
    Player,
)


@attr.s
class TeamMemberData(object):
    """ Information on a given team member going into a game. """
    player_id = attr.ib()  # The player id
    player_name = attr.ib()  # Player's name
    player_position = attr.ib()  # The player's position
    inactive = attr.ib(default=False)  # Whether or not this player is inactive for the game
    most_recent_player_season_date = attr.ib(default=None)  # The most recent PlayerSeasonDate, which would be
                                                            # the same one as the current game if this player
                                                            # plays in it, otherwise it's their previous game
    mins_recent_first = attr.ib(default=attr.Factory(list))  # The player's recent minutes as a list with most recent first
    mins_recent_weighted_average = attr.ib(default=0.0)  # The weighted average used to measure recent minutes played
    mins_average = attr.ib(default=0.0)  # The player's average minutes per game
    mins_metric = attr.ib(default=0.0)  # The metric used that combines recent and average statistics
    games_missed_recently = attr.ib(default=-1)  # How many consecutive games missed from this player going into
                                                 # this upcoming game
    max_mins_allowed = attr.ib(default=-1)  # The max amount of minutes this player is allowed to predicted for this game

    def is_capped(self):
        return np.isclose(self.mins_metric, self.max_mins_allowed)

    def first_position(self):
        return self.player_position.split('-')[0] if '-' in self.player_position else self.player_position

    def second_position(self):
        return self.player_position.split('-')[1] if '-' in self.player_position else self.player_position


def get_sorted_team_members(player_id_to_player_data, max_games_missed=3):
    """
    Sort the map of players by expected minutes to play (using "min_metric"). Also, exclude any
    players that are inactive and have missed more than "max_games_missed".

    :param player_id_to_player_map: map of player_id to their TeamMemberData
    :param max_games_missed: most games missed for an inactive player to be included in the
        returned map. The idea is players who have missed less than a certain amount of games
        should be considered for how to allocate their minutes elsewhere.
    :returns: A sorted mapping of player id to TeamMemberData based on mins_metric.
    """
    return {
        player_id: player_id_to_player_data[player_id]
        for player_id in [
            player_id for (player_id, data) in sorted(
                player_id_to_player_data.items(), key=lambda items: items[1].mins_metric, reverse=True
            )
        ]
        if not player_id_to_player_data[player_id].inactive
            or player_id_to_player_data[player_id].games_missed_recently <= max_games_missed
    }


def print_team_members(player_id_to_player_data):
    """ Print a debug message for a map of player id to TeamMemberData. """
    position_map = {
        'guard': 'G',
        'forward': 'F',
        'center': 'C',
        'forward-guard': 'F-G',
        'guard-forward': 'G-F',
        'forward-center': 'F-C',
        'center-forward': 'F-C',
        '': '',
        None: '',
    }
    print(
            "{:22s} {:6s} {:6s} {:3s}    {:5s}   {:5s}   {:5s}  {}"
        .format('Player', '', '', 'Out', '#', 'Rec', 'Avg', 'Last games')
    )
    print('\n{}'.format('-'*130))
    for team_member_id, team_member_data in player_id_to_player_data.items():
        print(
            "{:22s} {:6s} {:6s} {:3s}    {:5s}   {:5s}   {:5s}  {}"
            .format(
                team_member_data.player_name[:22] if team_member_data.player_name else team_member_id,
                position_map[team_member_data.player_position],
                "OUT" if team_member_data.inactive else "",
                str(team_member_data.games_missed_recently),
                str(round(team_member_data.mins_metric, 2)),
                str(round(team_member_data.mins_recent_weighted_average, 2)),
                str(round(team_member_data.mins_average, 2)),
                [round(num) if num is not None else None for num in team_member_data.mins_recent_first]
            )
        )


def get_multiplier(player_data):
    """
    Get a multiplier based on recent games missed. Specifically this is for active players coming
    back from injuries.
    """
    if not player_data.inactive and player_data.games_missed_recently >= 5:
        return 0.9
    return 1.0


def normalize_player_predictions(team_members_remaining, number_players_playing):
    """
    Normalize the top number_players_playing over 240 minutes, and zero predictions for the guys who are not
    expected to be playing.

    For any player:
      - Multiply the metric by a multiplier that takes into account recent games missed
      - Cap the prediction at a global max, and also at a max relative to their recent games

    :param team_memberes_remaining: map of player_id to TeamMemberData for each player
    :param number_players_paying: the number of guys who are expected to play in the game
    :returns: nothing, but normalizes the predictions in team_members_remaining
    """
    # Get a sum of the top players to play
    sum_top_players = sum(
        team_members_remaining[p].mins_metric for p in list(team_members_remaining.keys())[:number_players_playing]
    )

    # Iterate over top players to play and normalize their predictions to 240 for top players playing, and zero
    # it for the guys not prediction to be playing.
    for player_index, (player_id, player_data) in enumerate(team_members_remaining.items()):
        if player_index < number_players_playing:
            player_data.mins_metric = 240.0 * (
                get_multiplier(player_data) * player_data.mins_metric / sum_top_players
            )
            # Multiply this player's metric by a multiplier that takes into account recent games missed
            player_data.mins_metric *= get_multiplier(player_data)
            # Cap the prediction by the maximum allowed for this player
            player_data.mins_metric = min(player_data.mins_metric, player_data.max_mins_allowed)
        else:
            player_data.mins_metric = 0.0

    # Now, account for our post processing of normalization (the multiplier and the max-ing) creating a net total
    # greater than 240.0 minutes, which is the total we should end up with
    # TODO (JS): handle edge case of all capped players, otherwise infinite while loop (super edge case)
    while True:

        # If at 240 minutes predicted, we're done!
        total_predicted_mins = sum(
            team_member_data.mins_metric for team_member_data in team_members_remaining.values()
        )
        if not np.isclose(240.0, total_predicted_mins) and total_predicted_mins > 240.0:
            raise Exception("Predicted more than 240 normalized minutes, wtf, look into this!")
        if np.isclose(240.0, total_predicted_mins):
            break

        # Add up the amount of minutes of capped guys, and not capped guys, to re-normalize
        sum_capped_players = sum_not_capped_players = 0
        for team_member_data in team_members_remaining.values():
            if team_member_data.is_capped():
                sum_capped_players += team_member_data.mins_metric
            else:
                sum_not_capped_players += team_member_data.mins_metric

        # Re-normalize non-capped players relative to the non-capped minutes remaining. Don't re-apply the
        # multiplier here, since it was already applied once before.
        for player_id, player_data in team_members_remaining.items():
            player_data.mins_metric = (
                (240.0 - sum_capped_players) * (player_data.mins_metric / sum_not_capped_players)
            ) if not player_data.is_capped() else player_data.mins_metric
            # Re-check for a player who may have surprassed their allowed maximum
            player_data.mins_metric = min(player_data.mins_metric, player_data.max_mins_allowed)


def get_inactive_player_data(sorted_team_members, number_players_playing):
    """
    Get the inactive player data if there is an inactive player within the number_players_playing.

    :param sorted_team_members: map of player id to TeamMemberData sorted by mins_metric
    :param number_players_playing: number of guys to consider for an inactive guy
    :returns: TeamMemberData for inactive player, if one exists in top nunmber_players_playing, else None
    """
    for team_member_index, team_member_data in enumerate(
        list(sorted_team_members.values())[:number_players_playing]
    ):
        if team_member_data.inactive:
            return team_member_data
    return None


def find_player_to_add(players_to_consider, inactive_player_data):
    """
    Find the right player among "players_to_consider" to add to the lineup, to replace the "inactive_player_data".

    :returns: player data for the new player to add to lineup
    """
    def check_for_match(position):
        # First, check against the candidate player's first position
        for player_to_consider in players_to_consider:
            if position == player_to_consider.first_position():
                return player_to_consider

        # Next, check against the candidate player's second position
        for player_to_consider in players_to_consider:
            if position == player_to_consider.second_position():
                return player_to_consider
        return None

    # First, check for a match with the inactive player's first position
    candidate_player = check_for_match(inactive_player_data.first_position())
    if candidate_player:
        return candidate_player
    # Next, if inactive player has a different second position, try that
    if inactive_player_data.first_position != inactive_player_data.second_position():
        candidate_player = check_for_match(inactive_player_data.second_position())
        if candidate_player:
            return candidate_player
    # Next, if no players match the expected position(s), then just return the guy with the highest mins_metric
    return players_to_consider[0]


def update_lineup_for_inactive_player(
    team_members_remaining, number_players_playing, inactive_player_data, original_team_members
):
    """
    Update "team_members_remaining" based on the "inactive_player_data" being out, and expecting
    "number_players_playing" to play.

    The "original_team_members" is the original data going into the team predictions, which can be used
    to pull original data on a new guy who may be playing.

    :returns: Updates the team_members_remaining for the above.
    """

    # If there are no extra players left to add, delete the inactive guy and move on
    if len(team_members_remaining) <= number_players_playing:
        del team_members_remaining[inactive_player_data.player_id]
        return

    # Else, consider the players who aren't already marked as playing, and then delete the inactive player
    # from the team_members_remaining
    players_to_consider = list(team_members_remaining.values())[number_players_playing:]
    del team_members_remaining[inactive_player_data.player_id]

    # Find the right player to add to the lineup
    player_to_add = find_player_to_add(players_to_consider, inactive_player_data)

    # For now, just add the guy to the lineup with his original mins_metric
    team_members_remaining[
        player_to_add.player_id
    ].mins_metric = original_team_members[player_to_add.player_id].mins_metric


def get_team_game_min_predictions(
    team_members, number_players_playing, rotation_stats, max_player_mins=36.0, debug=True
):
    """
    Predict each player's minutes to be played in this game. This prediction is invariant of which
    player we're actually predicting for at the time, because we are predicting the full team's minutes
    at once for this game.

    :param team_members: A mapping of each team member on the team, to their TeamMemberData
    :param number_players_playing: The expectation of how many players this team will play in the game
    :param rotation_stats: The team's rotation stats up until this point in the season
    :param max_player_mins: The max minutes to allow for a player to be predicted for
    :returns: A mapping of each player to the minutes they're predicted to play in this game
    """

    sorted_team_members = get_sorted_team_members(team_members)
    if debug:
        print_team_members(sorted_team_members)

    # Update TeamMemberData for max allowable player minutes between a globa max, and the max of their recent games
    for player_id, player_data in sorted_team_members.items():
        player_data.max_mins_allowed = min(
            max_player_mins,
            # Add the 0.0 in case it's an otherwise empty sequence
            max([min_recent for min_recent in player_data.mins_recent_first if min_recent] + [0.0]),
        )

    # Iterate forever until the top number_players_playing includes no inactive players
    team_members_remaining = deepcopy(sorted_team_members)

    while True:

        # Normalize the top number_players_playing over 240 minutes, and zero predictions for the guys who are not
        # predicted to be playing. Note inactive players who remain so far, are not removed in this step.
        normalize_player_predictions(team_members_remaining, number_players_playing)

        print_team_members(team_members_remaining)

        # Find the first inactive player, sorted by most expected minutes. If the player is beyond the guys
        # expected to be playing, then we're done!
        inactive_player_data = get_inactive_player_data(team_members_remaining, number_players_playing)
        if inactive_player_data is None:
            break

        # Update players playing based on the inactive guy
        update_lineup_for_inactive_player(
            team_members_remaining, number_players_playing, inactive_player_data, sorted_team_members
        )

        # After the loop, re-sort the list of guys based on changes that may have happened
        team_members_remaining = get_sorted_team_members(team_members_remaining)
        print_team_members(team_members_remaining)

        import IPython; IPython.embed()















        @attr.s
        class MinsWithWithout(object):
            mins_with = attr.ib(default=0.0)
            mins_without = attr.ib(default=0.0)
            mins_total = attr.ib(default=0.0)

        # Get the player ids of the current guys to be playing, which we'll map to minutes with and without
        # the inactive player
        guys_playing = {
            player_id: MinsWithWithout() for player_id in list(team_members_remaining.keys())[:number_players_playing]
        }

        # Iterate through all lineups
        for rotation_ids, rotation_mins in rotation_stats.items():
            inactive_in_lineup = False
            all_players_valid = True
            for player_id in rotation_ids:
                if player_id == inactive_player_id:
                    inactive_in_lineup = True
                    continue
                elif player_id not in guys_playing:
                    all_players_valid = False
                    break
            if not all_players_valid:
                continue

            # Valid lineup (either with or without inactive dude)
            for player_id in rotation_ids:
                if inactive_in_lineup and player_id != inactive_player_id:
                    guys_playing[player_id].mins_with += rotation_mins
                    guys_playing[player_id].mins_total += rotation_mins
                elif not inactive_in_lineup:
                    guys_playing[player_id].mins_without += rotation_mins
                    guys_playing[player_id].mins_total += rotation_mins

        # TODO (JS): I AM HERE!! USE THIS TO DEBUG:
        print_debug()
        print("\n\n\nWithout      {} ({})\n".format(inactive_player.name, inactive_player.position))
        print("{:20s} {:8s} {:8s} {:10s}".format("Player", "With", "Without", "% Without"))
        for guy_playing_id, mins in guys_playing.items():
            player = Player.objects(pk=guy_playing_id).first()
            print("{:20s} {:8s} {:8s} {:10s} {}".format(player.name, str(round(mins.mins_with, 2)), str(round(mins.mins_without, 2)), str(round(mins.mins_without/mins.mins_total, 2) if mins.mins_total > 0 else 0), player.position))
        print("\n\n")
        ###########################################

        # Normalize each player relative to their expectation going into the game. If they've played less
        # than 10% of the relevant minutes of the most guy, assume the data is insufficient and go with a
        # standard default ratio
        most_mins_for_a_guy = max(guys_playing, key=lambda guy: guys_playing[guy].mins_total)
        for guy_playing_id, mins in guys_playing.items():
            expected_before_injury = sorted_team_members[guy_playing_id].mins_metric

        import IPython; IPython.embed()
        team_members_remaining = get_sorted_team_members(team_members_remaining)
        continue

        # Multiple the right column % by their expected minutes going in

        # Look at each other player and how many minutes they play with the inactive player, and figure out
        # a heuristic for how many more minutes they'll play without him
        extra_mins_map = {}
        for team_member_id, team_member_data in team_members_remaining.items():
            mins_with = total_mins = 0
            for lineup, lineup_mins in rotation_stats.items():
                if team_member_id in lineup:
                    total_mins += lineup_mins
                    if inactive_player_id in lineup:
                        mins_with += lineup_mins
            extra_mins_map[team_member_id] = (
                np.interp(mins_with/total_mins, (0.5, 0.75), (0.5, 0.0)) if total_mins > 0 else 0.5
            ) * team_member_data.mins_metric

        import IPython; IPython.embed()
        total_extra_mins = sum(extra_mins_map.values())
        for player_id, raw_extra_mins in extra_mins_map.items():
            player_data = team_members_remaining[player_id]
            extra_mins = inactive_mins_to_makeup * (raw_extra_mins / total_extra_mins)
            if extra_mins > (max_player_mins_map[player_id] - player_data.mins_metric):
                extra_mins = max_player_mins_map[player_id] - player_data.mins_metric
            player_data.mins_metric += extra_mins
            inactive_mins_to_makeup -= extra_mins
            total_extra_mins -= raw_extra_mins
        team_members_remaining = get_sorted_team_members(team_members_remaining)



    return {player_id: 0.0 for player_id in team_members_remaining.keys()}

    # TODO (JS): there still may be inactive guys if there's not enough totals
    import IPython; IPython.embed()

    # TODO (JS): my issue now is I'm giving too much to the shit players



    """ BELOW HERE IS TRASH FOR NOW, FIX ABOVE """


    ### Attempt ###
    team_members_remaining = deepcopy(sorted_team_members)
    # Iterate "forever" until the top number_players_playing includes no inactive players
    # TODO (JS): consider just ignoring guys who haven't played in >X games (i.e. more than 2 games), they'll effectively
    #            already be taken care of by the recent mins of other guys
    # TODO (JS): there's an issue here with current implementation because if a dude who plays 2 MINpg is inactive, I'm
    #            currently gonna probably multiple by the top guy by 1.99 (no shared minutes). I need to scale this by
    #            the amount of minutes the guy actually plays that's missing somehow
    # TODO (JS): I need to find a way to apply dame's 36 minutes as 36 minutes, not arbitrary factors that create fake
    #            minutes for scrubs
    while True:
        # Find the first player sorted by most expected minutes to be inactive
        inactive_player_id = inactive_player_index = inactive_player_data = None
        for candidate_team_member_index, (candidate_team_member_id, candidate_team_member_data) in enumerate(
            team_members_remaining.items()
        ):
            if candidate_team_member_data.inactive:
                inactive_player_id = candidate_team_member_id
                inactive_player_index = candidate_team_member_index
                inactive_player_data = candidate_team_member_data
                break
        # If we've found enough active players, or there are no more inactive players
        if inactive_player_index is None or inactive_player_index >= number_players_playing:
            break
        # Look at each other player and how many minutes they play with the inactive player, and update
        extra_mins_map = {}
        players_adjusted_for = 0
        for team_member_id, team_member_data in team_members_remaining.items():
            # Ignore the inactive player himself
            if team_member_id == inactive_player_id or team_member_data.inactive or players_adjusted_for >= number_players_playing:
                continue
            mins_with = total_mins = 0
            for lineup, lineup_mins in rotation_stats.items():
                if team_member_id in lineup:
                    total_mins += lineup_mins
                    if inactive_player_id in lineup:
                        mins_with += lineup_mins
            # Map from 50% of minutes shared to 50% bump in mins, to 75% of minutes shared to 0% bump
            extra_mins_map[team_member_id] = (
                np.interp(mins_with/total_mins, (0.5, 0.85), (0.5, 0.1)) if total_mins > 0 else 0.5
            ) * team_member_data.mins_metric
            players_adjusted_for += 1
            # team_member_data.mins_metric *= (1 + (1.0 - (mins_with/total_mins if total_mins > 0.0 else 0.0)))
        # Remove the inactive player now
        del team_members_remaining[inactive_player_id]
        total_extra_mins = sum(extra_mins_map.values())
        # TODO (JS): Do I wanna keep this intermediately?
        for other_player_id, other_player_extra_mins in extra_mins_map.items():
            team_members_remaining[other_player_id].mins_metric += (other_player_extra_mins / total_extra_mins) * inactive_player_data.mins_metric
        team_members_remaining = get_sorted_team_members(team_members_remaining)

    # It's possible there wasn't enough active players relative to number_players_playing, so I need to ensure all
    # inactive players are removed at this point
    team_members_remaining = {
        team_member_id: team_member_data for team_member_id, team_member_data in team_members_remaining.items()
        if not team_member_data.inactive
    }

    # Get a sum of the top players to play
    sum_top_players = sum(
        team_members_remaining[p].mins_metric for p in list(team_members_remaining.keys())[:number_players_playing]
    )

    # Create a multiplier based on recent games missed
    def get_multiplier(games_missed):
        if games_missed >= 5:
            return 0.9
        return 1.0

    # TODO (JS): I need to make sure my final result adds up to 240, which the maxing doesn't hold now

    # Take the min of the derived value, the player's recent max, and a global max
    player_map = {
        player_id: min(
            240.0 * (
                get_multiplier(player_data.games_missed_recently) *
                player_data.mins_metric / sum_top_players
            ),
            # Add the 0.0 in case t's an otherwise empty sequence
            max([min_recent for min_recent in player_data.mins_recent_first if min_recent] + [0.0]),
            36
        )
        if i < number_players_playing else 0.0
        for i, (player_id, player_data) in enumerate(team_members_remaining.items())
    }

    return player_map


def get_min_predictions(
    data_inputs, data_accounting, recent_lag=3, recent_to_be_considered=3, average_to_recent_ratio=0.5
):
    """
    Predict minutes played for a given game for each given player.

    :param data_inputs: The input data including any stats going into the game like MPG or recent MPG
    :param data_accounting: The accounting data for the given entries, including game id, player id, etc.
    :param recent_lag: The amount of recent games to consider for recent minutes played
    :param recent_to_be_considered: The amount of games ago to have played to be considered relevant
    :param average_to_recent_ratio: The percent to consider for average versus recent. For example 0.7 for
        this value means the calcuation for minutes metric would be:
          0.7 * average + 0.3 * recent
    :returns: the predicted minutes for each given instance of player in a game
    """

    # Make a new dataframe with all the data in one place, and a new column for predictions
    data = pd.concat((data_inputs, data_accounting), axis=1).copy()  # Copy just in case it's not implicitally
    data['MIN_PREDICTION'] = None

    # Cache mongo lookups so reduce lookups
    game_id_to_game = {}
    player_id_to_player_data = {}
    player_id_and_year_to_player_season = {}
    team_id_and_year_to_team_season = {}
    team_id_and_date_to_rotation_stats = {}
    team_id_and_game_id_to_predictions = {}

    data = data.sample(frac=1)

    # Iterate over all rows and predict for each one
    for player_index, player_game in data.iterrows():
        year = player_game.YEAR

        # Update cache for game and get game
        if player_game.GAME_ID not in game_id_to_game:
            game_id_to_game[player_game.GAME_ID] = Game.objects(game_id=player_game.GAME_ID).first()
        game = game_id_to_game[player_game.GAME_ID]

        # Update cache for team season and get team season
        if (player_game.TEAM_ID, year) not in team_id_and_year_to_team_season:
            team_id_and_year_to_team_season[
                (player_game.TEAM_ID, year)
            ] = TeamSeason.objects(team_id=player_game.TEAM_ID, year=year).first()
        team_season = team_id_and_year_to_team_season[(player_game.TEAM_ID, year)]

        # Find this player's team's stats going into the game, and the season index for this game.
        team_season_dates = [season_date.date for season_date in team_season.season_stats]
        team_season_index = bisect.bisect_right(team_season_dates, player_game.DATE) - 1
        team_season_date = team_season.season_stats[team_season_index]

        # Get how many players this team normally plays in a game
        number_players_playing = team_season_date.players_played_per_game

        # Update cache for rotation stats and get rotation stats
        if (player_game.TEAM_ID, player_game.DATE) not in team_id_and_date_to_rotation_stats:
            team_id_and_date_to_rotation_stats[(player_game.TEAM_ID, player_game.DATE)] = {
                tuple(rotation.players): rotation.minutes[team_season_index] for rotation in team_season.rotation_stats
                if rotation.minutes[team_season_index] > 0.0
            }
        rotation_stats = team_id_and_date_to_rotation_stats[(player_game.TEAM_ID, player_game.DATE)]

        # Update cache for player seasons and add team members for:
        #  1) the player himself
        #  2) the inactive players
        #  3) the other players who played in this game on the same team as the player
        team_members = {}
        for player_id in [player_game.PLAYER_ID] + game.team_games[player_game.TEAM_ID].inactives + [
            other_player_id for other_player_id, other_player_game in game.player_games.items()
            if other_player_game.team_id == player_game.TEAM_ID and other_player_id != player_game.PLAYER_ID
        ]:

            # Update cache for player data
            if player_id not in player_id_to_player_data:
                player_id_to_player_data[player_id] = Player.objects(pk=player_id).first()
            player_data = player_id_to_player_data[player_id]

            team_members[player_id] = TeamMemberData(
                player_id=player_id,
                player_name=player_data.name if player_data else None,
                player_position=player_data.position if player_data else None,
            )
            if (player_id, year) not in player_id_and_year_to_player_season:
                player_id_and_year_to_player_season[
                    (player_id, year)
                ] = PlayerSeason.objects(player_id=player_id, year=year).first()
        for inactive_id in game.team_games[player_game.TEAM_ID].inactives:
            team_members[inactive_id].inactive = True

        # For each team member, fill in the relevant data fields
        for team_member_id, team_member_data in team_members.items():

            # Get the player's season
            player_season = player_id_and_year_to_player_season[(team_member_id, year)]
            if not player_season:
                continue

            # Get 3 things:
            #  1) "most_recent_season_data": which represents the most recent set of stats for this player
            #                                up until this game
            #  2) "date_of_last_game_played": the date of the game last played by this player before the
            #                                 current game
            #  3) "player_season_index_of_last_game_played": the season index of the player season from the
            #                                                last game played by this player before the
            #                                                current game
            # Get the game corresponding to the current game for this player's season, or if they
            # didn't play in this game, get the most recently played game before this game for this player
            most_recent_season_date = None
            date_of_last_game_played = None
            player_season_index_of_last_game_played = None
            for player_season_index, season_date in enumerate(player_season.season_stats):
                # If this date is before the current date, update both
                if season_date.date < player_game.DATE:
                    most_recent_season_date = season_date
                    date_of_last_game_played = season_date.date
                    player_season_index_of_last_game_played = player_season_index
                # Else if this date is the same day as the current date, only update the
                # most_recent_season_date, since that represents that stats going into this game. But, we
                # don't want to update "date_of_last_game_played" because this IS the date of the current game
                elif season_date.date == player_game.DATE:
                    most_recent_season_date = season_date
                # Else we have passed the current game, break
                else:
                    break
            if (
                most_recent_season_date is None or most_recent_season_date.stats is None
                or date_of_last_game_played is None
            ):
                continue
            team_member_data.most_recent_player_season_date = most_recent_season_date

            # Store the player's average and recent minutes for this season thus far
            team_member_data.mins_average = most_recent_season_date.stats.per_game.MIN
            team_member_data.mins_recent_first = most_recent_season_date.stats.recent.MIN_RECENT_FIRST

            # Calculate and store the weighted average for recent minutes played
            numerator = denominator = 0
            for i in range(recent_lag):
                if team_member_data.mins_recent_first[i] is None:
                     continue

                """
                # Incorporate checking for blowouts, this seemed to not help?
                #############################################################
                recent_game_id = player_season.season_stats[player_season_index_of_last_game_played - i].game_id
                if recent_game_id not in game_id_to_game:
                    game_id_to_game[recent_game_id] = Game.objects(game_id=recent_game_id).first()
                recent_game = game_id_to_game[recent_game_id]
                score_difference = np.abs(
                    np.diff(
                        [team_game.traditional_stats.PTS for team_game in recent_game.team_games.values()]
                    )[0]
                )
                scaler = np.interp(score_difference, [18, 30], [1.0, 0.0])
                numerator += team_member_data.mins_recent_first[i] * (recent_lag - i) * scaler
                denominator += (recent_lag - i) * scaler
                #############################################################
                """
                numerator += team_member_data.mins_recent_first[i] * (recent_lag - i)
                denominator += (recent_lag - i)
            team_member_data.mins_recent_weighted_average = numerator/denominator if denominator > 0 else 0.0

            # Calculate and store the mins metric
            team_member_data.mins_metric = (
                average_to_recent_ratio * team_member_data.mins_average
                + ((1 - average_to_recent_ratio) * team_member_data.mins_recent_weighted_average)
            )

            # Find and store how many games this player has missed consecutively before this game. Subtract one
            # from the calculation because there should be a 1 game difference between games if no games were missed.
            last_game_played_team_season_index = bisect.bisect_right(team_season_dates, date_of_last_game_played) - 1
            team_member_data.games_missed_recently = team_season_index - last_game_played_team_season_index - 1

        # Predict minutes per player for this team in this game. These values are calculate per team-game, so
        # we can cache the results since the results apply regardless of who we're looking at.
        if player_game.DATE >= '2022-01-03':
        # if player_game.TEAM_ID == '1610612766':
            print("DATE: {}\n\n".format(player_game.DATE))
            if (player_game.GAME_ID, player_game.TEAM_ID) not in team_id_and_game_id_to_predictions:
                team_id_and_game_id_to_predictions[
                    (player_game.GAME_ID, player_game.TEAM_ID)
                ] = get_team_game_min_predictions(
                    team_members=team_members,
                    number_players_playing=number_players_playing,
                    rotation_stats=rotation_stats,
                )
            data.at[player_index, 'MIN_PREDICTION'] = team_id_and_game_id_to_predictions[
                (player_game.GAME_ID, player_game.TEAM_ID)
            ][player_game.PLAYER_ID]


        ### TODO ###
        # 1) There should be a cap, for example if a lot of guys are out, the dudes with high mins pg already
        #    may go from recent of like 33 to 42, which isn't reasonable. Need to propogate this downwards to other
        #    dudes
        #      - This can either be max for the given player (like his recent max)
        #      - Max for any player (like 38)
        #      - mix of both
        # 2) Can take into account who is out better to more explicitly propogate available minutes
        # 3) Probably makes sense to consider one (game, team) once and cache, since we're calculating all players
        #    expectations at once, since they affect each other
        # 4) Probably need more accurate way to remove players who aren't playing
        # 5) If someone hasn't played in a while, can decrease his minutes probably for some ratio of "return"
        # 6) When calculating a recent stat, if game is a blowout.. maybe don't count the game or count it less?
        #    A dude that plays 30 mins in a 30 point game isn't really indicative of his minutes in next game
        # 7) Need to figure out how to minimize contributions of lowly dudes. If random dude X plays 1 recent game
        #    gonna show up in recents with 30 mins each for recent and will skew the average away from the guys
        #    who matter.
        #       - The trouble in resolving this is a lot of thoughts involve differentiating between a dude like
        #         this not playing, and real guys being inactive for injuries
        # 8) I need to separate the guys who didnt play because they were inactive/hurt, and those who didn't
        #    play because they suck. The ones that dont play because they suck should have a 0 for those games
        #    to weight their averages better, instead of just a random "20 min" game which gives them what looks
        #    like a consistent 20 MPG
        # 9) I think I have a bug where if someone hasn't played for 4 games, they won't be counted? i.e. a player
        #    returning from actual injury? Also related, is that if a player hasn't played for consecutive games
        #    (meaning not just rest), maybe should dock him to like 75% his average or some ish.
        # 10) I think recent_to_be_considered is invalid and should be removed. Instead, I need to consider all
        #     guys because CJ McCollum may be out for 1 month, but when he comes back he's still gonna play 28
        #     minutes. But I need to have a way to differentiate from a scrub who doesn't play for a month and
        #     then plays. I think I can use average to decipher this:
        #         - Use recents, have some consideration of when they haven't played for some time
        #         - Use average as a metric as to whether or not they should have high expectations even after
        #           not playing
        #         - Probably still apply some down percentage on guys who were out for a while (first game back)
        # 11) KEY!! More generally, I think I need to factor in average to recent as well (maybe weighted average).
        #     I can use this to downrank guys who had random high min games but overall rarely play that much.
        # 12) Consider back to backs (historically, and also for the given upcoming prediction)
        ############


    return data[['MIN_PREDICTION']]


