import pandas as pd
import bisect
import attr

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


def get_team_game_min_predictions(team_members, number_players_playing, rotation_stats, debug=False):
    """
    Predict each player's minutes to be played in this game. This prediction is invariant of which
    player we're actually predicting for at the time, because we are predicting the full team's minutes
    at once for this game.

    :param team_members: A mapping of each team member on the team, to their TeamMemberData
    :param number_players_playing: The expectation of how many players this team will play in the game
    :param rotation_stats: The team's rotation stats up until this point in the season
    :returns: A mapping of each player to the minutes they're predicted to play in this game
    """

    # Sort the players in terms of their recent minutes numbers
    sorted_team_members = {
        player_id: team_members[player_id]
        for player_id in [
            player_id for (player_id, data) in sorted(
                team_members.items(), key=lambda items: items[1].mins_metric, reverse=True
            )
        ]
    }

    if debug:
        print(
            "{:20s} {:10s} {:3s}    {:5s}   {:5s}   {:5s}  {}"
            .format('Player', '', 'Out', '#', 'Rec', 'Avg', 'Last games')
        )
        print('\n{}'.format('-'*130))
        for team_member_id, team_member_data in sorted_team_members.items():
            player_object = Player.objects(pk=team_member_id).first()
            print(
                "{:22s} {:10s} {:3s}    {:5s}   {:5s}   {:5s}  {}"
                .format(
                    player_object.name if player_object else team_member_id,
                    "Inactive" if team_member_data.inactive else "",
                    str(team_member_data.games_missed_recently),
                    str(round(team_member_data.mins_metric, 2)),
                    str(round(team_member_data.mins_recent_weighted_average, 2)),
                    str(round(team_member_data.mins_average, 2)),
                    team_member_data.mins_recent_first
                )
            )

    ### Temp solution to take sum of top number players playing and normalize ###
    sorted_team_members_playing = {
        player_id: player_data for player_id, player_data in sorted_team_members.items()
        if not player_data.inactive
    }
    sum_top_players = sum(
        sorted_team_members_playing[p].mins_metric for p in list(sorted_team_members_playing.keys())[:number_players_playing]
    )
    def get_multiplier(games_missed):
        if games_missed >= 5:
            return 0.9
        return 1.0
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
        for i, (player_id, player_data) in enumerate(sorted_team_members_playing.items())
    }
    return player_map
    ################################################################################


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
    player_id_and_year_to_player_season = {}
    team_id_and_year_to_team_season = {}
    team_id_and_date_to_rotation_stats = {}
    team_id_and_game_id_to_predictions = {}

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
        season_index = bisect.bisect_right(team_season_dates, player_game.DATE) - 1
        team_season_date = team_season.season_stats[season_index]

        # Get how many players this team normally plays in a game
        number_players_playing = team_season_date.players_played_per_game

        # Update cache for rotation stats and get rotation stats
        if (player_game.TEAM_ID, player_game.DATE) not in team_id_and_date_to_rotation_stats:
            team_id_and_date_to_rotation_stats[(player_game.TEAM_ID, player_game.DATE)] = {
                tuple(rotation.players): rotation.minutes[season_index] for rotation in team_season.rotation_stats
                if rotation.minutes[season_index] > 0.0
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
            team_members[player_id] = TeamMemberData(player_id=player_id)
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

            # Get 2 things:
            #  1) "most_recent_season_data": which represents the most recent set of stats for this player
            #                                up until this game
            #  2) "date_of_last_game_played": the date of the game last played by this player before the
            #                                 current game
            # Get the game corresponding to the current game for this player's season, or if they
            # didn't play in this game, get the most recently played game before this game for this player
            most_recent_season_date = None
            date_of_last_game_played = None
            for season_date in player_season.season_stats:
                # If this date is before the current date, update both
                if season_date.date < player_game.DATE:
                    most_recent_season_date = season_date
                    date_of_last_game_played = season_date.date
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
            last_game_played_season_index = bisect.bisect_right(team_season_dates, date_of_last_game_played) - 1
            team_member_data.games_missed_recently = season_index - last_game_played_season_index - 1

        # Predict minutes per player for this team in this game. These values are calculate per team-game, so
        # we can cache the results since the results apply regardless of who we're looking at.
        # if player_game.DATE >= '2022-01-03':
        # if player_game.TEAM_ID == '1610612766':
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
        ############


    return data[['MIN_PREDICTION']]


