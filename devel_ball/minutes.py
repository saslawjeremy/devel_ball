import pandas as pd
import bisect
import attr
import numpy as np
from copy import deepcopy
from datetime import datetime
from colorama import (
    Fore,
    Back,
    Style,
)

from devel_ball.models import (
    Game,
    TeamSeason,
    PlayerSeason,
    Player,
)



# 1) Remove all inactive players first, before adjusting for each one that was removed
# 2) smarter maxes, initial max of 36 but then allow guys to be higher based on circumstance
# 3) return back to adjusting for guys who are out



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
    back_to_back = attr.ib(default=False)  # Whether or not this player is playing on a back to back

    def is_capped(self):
        return np.isclose(self.mins_metric, self.max_mins_allowed)

    def first_position(self):
        if self.player_position is None:
            return None
        return self.player_position.split('-')[0] if '-' in self.player_position else self.player_position

    def second_position(self):
        if self.player_position is None:
            return None
        return self.player_position.split('-')[1] if '-' in self.player_position else self.player_position


@attr.s
class MinsWithWithout(object):
    """ Used to track minutes a player plays with and without another given player. """
    mins_with = attr.ib(default=0.0)
    mins_without = attr.ib(default=0.0)
    mins_total = attr.ib(default=0.0)


def get_sorted_team_members(player_id_to_player_data, max_games_missed=0):
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
    print('{}'.format('-'*130))
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
    if not player_data.inactive and player_data.games_missed_recently >= 7:
        return 0.9
    return 1.0


def normalize_player_predictions(team_members_remaining, number_players_playing, hard_max_mins):
    """
    Normalize the top number_players_playing over 240 minutes, and zero predictions for the guys who are not
    expected to be playing.

    For any player:
      - Multiply the metric by a multiplier that takes into account recent games missed
      - Cap the prediction at a global max, and also at a max relative to their recent games

    :param team_memberes_remaining: map of player_id to TeamMemberData for each player
    :param number_players_paying: the number of guys who are expected to play in the game
    :param hard_max_mins: the hard max on how many minutes a player can reach
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

        # If all players have reached a cap, we need to increase the max's to keep getting towards a solution
        if sum_not_capped_players == 0:
            for team_member_data in team_members_remaining.values():
                team_member_data.max_mins_allowed = min(team_member_data.max_mins_allowed * 1.05, hard_max_mins)
            sum_not_capped_players = sum_capped_players
            sum_capped_players = 0

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


def update_active_players_for_inactive_player(
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
        return team_members_remaining

    # Else, consider the players who aren't already marked as playing who aren't inactive, and then delete the
    # inactive player from the team_members_remaining
    players_to_consider = list(team_members_remaining.values())[number_players_playing:]
    del team_members_remaining[inactive_player_data.player_id]

    # Find the right player to add to the lineup
    player_to_add = find_player_to_add(players_to_consider, inactive_player_data)

    # Just add the guy to the lineup with his original mins_metric, but at least 1.0 mins to bound it
    new_mins = max(
        min(
            original_team_members[player_to_add.player_id].mins_metric,
            original_team_members[player_to_add.player_id].max_mins_allowed,
        ),
        1.0
    )
    # Make sure we're not making this guy have more minutes than the inactive guy
    new_mins = min(new_mins, inactive_player_data.mins_metric)
    team_members_remaining[player_to_add.player_id].mins_metric = new_mins

    # And subtract the amount we're adding for this
    inactive_player_data.mins_metric -= new_mins

    # Need to re-sort since the "next" guy available may not be the one we added.
    return get_sorted_team_members(team_members_remaining)


def get_relationship_map(team_member_ids_playing, rotation_stats, inactive_player_id):
    """
    Get a map of player_id of guy playing, to "MinsWithWithout" metrics on that player's relationship to
    the inactive player.

    :param team_member_ids_playing: a list of the team member id's who are playing
    :param rotation_stats: the rotation stats for this game
    :param inactive_player_id: the player id of the inactive player

    :returns: a map of player_id to MinsWithWithout for each guy playing in the game
    """

    relationship_map = {player_id: MinsWithWithout() for player_id in team_member_ids_playing}

    """
    # Strategy 1: Consider only "fully" valid lineups, proportion of minutes based on other guy's minutes played
    # Iterate through all rotations
    for rotation_ids, rotation_mins in rotation_stats.items():
        inactive_in_lineup = False
        all_players_valid = True
        for player_id in rotation_ids:
            if player_id == inactive_player_id:
                inactive_in_lineup = True
            elif player_id not in relationship_map:
                all_players_valid = False
                break
        if not all_players_valid:
            continue

        # Valid lineup (either with or without inactive dude)
        for player_id in rotation_ids:
            if inactive_in_lineup and player_id != inactive_player_id:
                relationship_map[player_id].mins_with += rotation_mins
                relationship_map[player_id].mins_total += rotation_mins
            elif not inactive_in_lineup:
                relationship_map[player_id].mins_without += rotation_mins
                relationship_map[player_id].mins_total += rotation_mins

    # Strategy 2: Proportion of minutes based on other guy's minutes played
    for rotation_ids, rotation_mins in rotation_stats.items():
        inactive_in_lineup = inactive_player_id in rotation_ids
        for player_id in rotation_ids:
            if player_id in relationship_map:
                if inactive_in_lineup:
                    relationship_map[player_id].mins_with += rotation_mins
                else:
                    relationship_map[player_id].mins_without += rotation_mins
                relationship_map[player_id].mins_total += rotation_mins
    """

    # Strategy 3: Proportion of minutes based on inactive guy's minutes played
    for rotation_ids, rotation_mins in rotation_stats.items():
        if inactive_player_id not in rotation_ids:
            continue
        for other_player_id, other_player_relationship in relationship_map.items():
            if other_player_id in rotation_ids:
                other_player_relationship.mins_with += rotation_mins
            else:
                other_player_relationship.mins_without += rotation_mins
            other_player_relationship.mins_total += rotation_mins

    return relationship_map


def get_replacement_tiers(team_members_data, inactive_player_data):
    """
    Put the other players in buckets of if they match the inactive player

    By 1st priority:
     - position 1 matches position 1
    By 2nd priority:
     - position 1 matches position 2
     - position 2 matches position 2
    By 3rd priority:
     - position 2 matches position 2
    By 4th priority:
     - no position matches

    :param team_member_data: a list of TeamMemberData for other players to consider
    :param inactive_player_data: the TeamMemberData for the inactive player

    :returns: a list of 4 lists, according to the above tier structure
    """
    tiers = [[], [], [], []]
    for team_member_data in team_members_data:
        if inactive_player_data.first_position() == team_member_data.first_position():
            tiers[0].append(team_member_data)
        elif inactive_player_data.first_position() == team_member_data.second_position():
            tiers[0].append(team_member_data)
        elif inactive_player_data.second_position() == team_member_data.first_position():
            tiers[0].append(team_member_data)
        elif inactive_player_data.second_position() == team_member_data.second_position():
            tiers[0].append(team_member_data)
        else:
            tiers[0].append(team_member_data)
    return tiers


def update_replacements(mins_left, replacements_metrics, team_members_remaining):
    """
    Update the replacements minutes based on the inactive player that is out.

    :param mins_left: The amount of minutes left to allocate
    :param replacements_metrics: A map of player_id to ReplacementMetrics
    :param team_members_remaining: map of player id to TeamMemberData of guys still remaining

    :returns: The amount of minutes still left after this update
    """
    still_going = True
    while not np.isclose(mins_left, 0.0) and mins_left > 0.0 and still_going:
        total_replacement_value = sum(
            replacement_metrics.replacement_value
            for replacement_id, replacement_metrics in replacements_metrics.items()
            if not np.isclose(team_members_remaining[replacement_id].mins_metric, replacement_metrics.max_mins)
        )

        still_going = False
        for replacement_id, replacement_metrics in replacements_metrics.items():
            replacement_data = team_members_remaining[replacement_id]
            if np.isclose(replacement_data.mins_metric, replacement_metrics.max_mins):
                continue
            still_going = True
            new_mins = min(
                replacement_metrics.max_mins,
                replacement_data.mins_metric + mins_left * (
                    replacement_metrics.replacement_value / total_replacement_value
                )
            )
            mins_left -= (new_mins - replacement_data.mins_metric)
            replacement_data.mins_metric = new_mins
    return mins_left


## TODO (JS): at 2.0, max bump can probably be removed all together
def update_player_mins_for_inactive_player(
    team_members_remaining, inactive_player_data, number_players_playing, rotation_stats, hard_max_mins, max_bump=2.0,
):
    """
    Update "team_members_remaining" based on the player being out described by "inactive_player_data".

    :param team_members_remaining: map of player id to TeamMemberData of guys still remaining
    :param inactive_player_data: TeamMemberData for the player who is out
    :param number_players_playing: int for how many players should be marked as playing
    :param rotation_stats: the team's RotationStats going into this game
    :param max_bump: the largest multiplier of minutes to bump a candidate replacement for the inactive player, for
        example a "max_bump" of 1.5 and a "mins_metric" of 12 means that player can only be allocated up to
        18 minutes in this step.

    :returns: an updated "team_members_remaining" with adjusted minutes
    """

    print(Back.RED + "Without      {} ({})".format(inactive_player_data.player_name, inactive_player_data.player_position) + Style.RESET_ALL)

    relationship_map = get_relationship_map(
        list(team_members_remaining.keys())[:number_players_playing], rotation_stats, inactive_player_data.player_id
    )

    # Get a list of 4 lists that puts the potential replacement players into tiers of priority to replace
    # the inactive guy, prioritized by positional matches
    replacement_tiers = get_replacement_tiers(
        list(team_members_remaining.values())[:number_players_playing], inactive_player_data
    )

    @attr.s
    class ReplacementMetrics(object):
        # Max mins for the replacement, by considering:
        #  - The global max this player is allowed
        #  - The current estimate for this player * "max_bump"
        max_mins = attr.ib()
        # The difference between the player who's out minutes expected, and the replacement's minutes expected
        mins_difference = attr.ib()
        # The percentage of minutes the replacement played without the injured player on the court
        mins_without_percentage = attr.ib()
        # An arbitrary heuristic at how much weighting to give to this replacement option. This is used by
        # summing all replacement options, and checking for each player's contribution to it, correpsonding to
        # their relevance as a replacement option.
        replacement_value = attr.ib(default=0.0)

        def __attrs_post_init__(self):
            self.replacement_value = np.interp(self.mins_without_percentage, (0.0, 1.0), (0.01, 0.99))
            # self.replacement_value = np.interp(self.mins_difference, (0, 20), (1.0, 0.0))

    # Iterate while there are minutes left to allocate
    mins_left = inactive_player_data.mins_metric
    max_multiplier = 1.0
    while True:

        # Iterate over each tier, applying minutes to players within that tier
        for replacement_tier in replacement_tiers:

            # If there's no more minutes to distribute, we're done
            if np.isclose(mins_left, 0) or mins_left <= 0.0:
                return get_sorted_team_members(team_members_remaining)

            # Gather the stats on the potental replacements
            replacements_metrics = {
                replacement.player_id: ReplacementMetrics(
                    max_mins=min(
                        min(replacement.max_mins_allowed, replacement.mins_metric * max_bump) * max_multiplier,
                        hard_max_mins,
                    ),
                    mins_difference=np.abs(replacement.mins_metric - inactive_player_data.mins_metric),
                    mins_without_percentage=
                        relationship_map[replacement.player_id].mins_without/
                            relationship_map[replacement.player_id].mins_total
                            if relationship_map[replacement.player_id].mins_total > 0 else 0.0,
                ) for replacement in replacement_tier
            }

            # Update the replacements in this tier, keeping track of how many minutes are still left after that
            mins_left = update_replacements(mins_left, replacements_metrics, team_members_remaining)

        # Allow for higher max's next time around
        max_multiplier += 0.1

def get_team_game_min_predictions(
    team_members,
    number_players_playing,
    rotation_stats,
    soft_max_mins=34.0,
    hard_max_mins=40.0,
    recent_games_average_exceeds_soft_max=4,
    debug=True
):
    """
    Predict each player's minutes to be played in this game. This prediction is invariant of which
    player we're actually predicting for at the time, because we are predicting the full team's minutes
    at once for this game.

    :param team_members: A mapping of each team member on the team, to their TeamMemberData
    :param number_players_playing: The expectation of how many players this team will play in the game
    :param rotation_stats: The team's rotation stats up until this point in the season
    :param soft_max_mins: The initial max minutes to nominally allow for a player to be predicted for,
        barring exceptional circumstances to suggest player may exceed this
    :param hard_max_mins: The hard max on minutes a player will ever be allowed to reach
    :param recent_games_average_exceeds_soft_max: The amount of recent games to average the minutes played in, where
        if the result is higher than the soft_max_mins, replace the soft_max_mins with this number
    :returns: A mapping of each player to the minutes they're predicted to play in this game
    """

    # TODO (JS): ideas:
    #  4) Account for OT night before

    sorted_team_members = get_sorted_team_members(team_members)
    if debug:
        print(Back.RED + "Original" + Style.RESET_ALL + "\n")
        print_team_members(sorted_team_members)

    # Update TeamMemberData for max allowable minutes played (ensuring at least 1.0) by taking the least of:
    #   1) The maximum of soft_max_mins, and the player's average minutes over a specified recent stretch of games
    #   2) The maximum this player has played in their recent games
    #   3) A hard max
    for player_id, player_data in sorted_team_members.items():
        player_data.max_mins_allowed = max(
            min(
                max(
                    soft_max_mins,
                    np.mean([p for p in player_data.mins_recent_first if p][:recent_games_average_exceeds_soft_max]),
                ),
                # Add the 0.0 in case it's an otherwise empty sequence
                max([min_recent for min_recent in player_data.mins_recent_first if min_recent] + [0.0]),
                hard_max_mins,
            ),
            1.0
        )

    # Normalize the top number_players_playing over 240 minutes, and zero predictions for the guys who are not
    # predicted to be playing. Note within the top "number_players_playing", inactive players may still remain,
    # but any players not in the top "number_players_playing" who are inactive, are removed at this time.
    team_members_remaining = deepcopy(sorted_team_members)
    normalize_player_predictions(team_members_remaining, number_players_playing, hard_max_mins)
    team_members_remaining = get_sorted_team_members(
        {
            player: player_data for i, (player, player_data) in enumerate(team_members_remaining.items())
            if i < number_players_playing or not player_data.inactive
        }
    )

    # TODO (JS): Go through helpers and test/tune any parameters of interest. If parameters truly change stuff,
    # then expose them to higher level caller to pass in.

    # Pull out all the inactive players, and replace them
    inactive_players = []
    while True:

        # Find the first inactive player, sorted by most expected minutes. If the player is beyond the guys
        # expected to be playing, then this step is done.
        inactive_player_data = get_inactive_player_data(team_members_remaining, number_players_playing)
        if inactive_player_data is None:
            break
        inactive_players.append(inactive_player_data)

        # Update players playing based on the inactive guy
        team_members_remaining = update_active_players_for_inactive_player(
            team_members_remaining, number_players_playing, inactive_player_data, sorted_team_members
        )

    # Go through the inactives, and apply their minutes to others in the rotation
    for inactive_player in inactive_players:

        # Intelligently update the other players in the lineup based on the inactive player
        team_members_remaining = update_player_mins_for_inactive_player(
            team_members_remaining, inactive_player, number_players_playing, rotation_stats, hard_max_mins
        )

    assert(np.isclose(sum(p.mins_metric for p in team_members_remaining.values()), 240))

    if debug:
        print_team_members(team_members_remaining)
    # Safety check that hard max isn't violated
    assert(max(data.mins_metric for data in team_members_remaining.values()) <= hard_max_mins)
    return {player_id: data.mins_metric for player_id, data in team_members_remaining.items()}


def get_min_predictions(data_inputs, data_accounting, recent_lag=5, average_to_recent_ratio=0.5):
    """
    Predict minutes played for a given game for each given player.

    :param data_inputs: The input data including any stats going into the game like MPG or recent MPG
    :param data_accounting: The accounting data for the given entries, including game id, player id, etc.
    :param recent_lag: The amount of recent games to consider for recent minutes played
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

    # Randomize
    # data = data.sample(frac=1)

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
            #  1) "most_recent_season_date": which represents the most recent set of stats for this player
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

                # If this is a blowout, don't count the game
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
                if score_difference >= 35:
                    continue
                #############################################################
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

            # Store if this player is playing on a back to back. Note the first number in the "mins_recent_first"
            # list represents the amount of minutes played in this last game
            team_member_data.back_to_back = (
                datetime.strptime(player_game.DATE, "%Y-%m-%d")
                - datetime.strptime(date_of_last_game_played, "%Y-%m-%d")
            ).days == 1

        # Predict minutes per player for this team in this game. These values are calculate per team-game, so
        # we can cache the results since the results apply regardless of who we're looking at.
        # if player_game.DATE <= '2022-01-03':
        # if player_game.TEAM_ID == '1610612766':
        print("\n\n" + Back.GREEN + Fore.BLACK + "DATE: {}".format(player_game.DATE) + Style.RESET_ALL + "\n\n")
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
        # 7) Need to figure out how to minimize contributions of lowly dudes. If random dude X plays 1 recent game
        #    gonna show up in recents with 30 mins each for recent and will skew the average away from the guys
        #    who matter.
        #       - The trouble in resolving this is a lot of thoughts involve differentiating between a dude like
        #         this not playing, and real guys being inactive for injuries
        ############

    return data[['MIN_PREDICTION']]


