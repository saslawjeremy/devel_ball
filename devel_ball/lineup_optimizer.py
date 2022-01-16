import pulp
import numpy as np

from devel_ball.analysis import predict_from_model


def get_sorted_players(prediction_map, print_option=None):

    # Sort the predictions by total points
    sorted_predictions_by_total_points = dict(
        sorted(prediction_map.items(), key=lambda item: -item[1][0])
    )
    # Sort the predictions by value (points per cost)
    sorted_predictions_by_value = dict(
        sorted(prediction_map.items(), key=lambda item: -item[1][0] / item[1][1].cost)
    )

    # Print one of them if the option is provided
    def print_helper(sorted_predictions):
        for player, (prediction, dk_player) in sorted_predictions.items():
            print(
                "{}, prediction: {}, cost: {}, value: {}".format(
                    player, prediction, dk_player.cost, round(prediction/dk_player.cost*1000, 3)
                )
            )
    if print_option == 'POINTS':
        print_helper(sorted_predictions_by_total_points)
    elif print_option == 'VALUE':
        print_helper(sorted_predictions_by_value)

    return sorted_predictions_by_total_points, sorted_predictions_by_value


def run_lineup_optimizer(prediction_map, players, must_include_players):
    """
    Take prediction map of players to predicted values, and a list of players to consider,
    a subset of those players that must be added, and output the most desirable lineup
    considering lineup constraints.
    """

    # Ensure players specified to be included are valid options
    for must_include_player in must_include_players:
        if must_include_player not in players:
            raise Exception("Cannot find {} to include.".format(must_include_player))

    # Create mappings used for linear programming optimizer
    points = {player: prediction_map[player][0] for player in players}
    costs = {player: prediction_map[player][1].cost for player in players}
    pgs = {player: 1 if "PG" in prediction_map[player][1].positions else 0 for player in players}
    sgs = {player: 1 if "SG" in prediction_map[player][1].positions else 0 for player in players}
    sfs = {player: 1 if "SF" in prediction_map[player][1].positions else 0 for player in players}
    pfs = {player: 1 if "PF" in prediction_map[player][1].positions else 0 for player in players}
    cs = {player: 1 if "C" in prediction_map[player][1].positions else 0 for player in players}
    gs = {player: pgs[player] or sgs[player] for player in players}
    fs = {player: sfs[player] or pfs[player] for player in players}
    g_or_fs = {player: gs[player] or fs[player] for player in players}

    # Create linear programming problem and add constraints, and then solve
    prob = pulp.LpProblem("Draftkings", pulp.LpMaximize)
    player_vars = pulp.LpVariable.dicts("Players", players, 0, 1, pulp.LpBinary)
    prob += pulp.lpSum([points[player] * player_vars[player] for player in players]), "Total Cost"
    prob += pulp.lpSum([player_vars[player] for player in players]) == 8, "Total 8 Players"
    prob += pulp.lpSum([pgs[player] * player_vars[player] for player in players]) >= 1, "At least 1 PG"
    prob += pulp.lpSum([sgs[player] * player_vars[player] for player in players]) >= 1, "At least 1 SG"
    prob += pulp.lpSum([sfs[player] * player_vars[player] for player in players]) >= 1, "At least 1 SF"
    prob += pulp.lpSum([pfs[player] * player_vars[player] for player in players]) >= 1, "At least 1 PF"
    prob += pulp.lpSum([cs[player] * player_vars[player] for player in players]) >= 1, "At least 1 C"
    prob += pulp.lpSum([gs[player] * player_vars[player] for player in players]) >= 3, "At least 3 G"
    prob += pulp.lpSum([fs[player] * player_vars[player] for player in players]) >= 3, "At least 3 F"
    prob += pulp.lpSum([g_or_fs[player] * player_vars[player] for player in players]) >= 6, "At least 6 G/Fs"
    prob += pulp.lpSum([costs[player] * player_vars[player] for player in players]) <= 50000, "Total Cost"
    prob += pulp.lpSum(
        [player_vars[player] for player in must_include_players]
    ) == len(must_include_players), "Must include players"
    status = prob.solve()

    if status == -1:
        raise Exception("Could not optimize a lineup!")

    # Get total points/cost, and players chosen
    total_points = 0.0
    total_costs = 0.0
    chosen_players = []
    for player, player_var in player_vars.items():
        if player_var.varValue > 0:
            total_points += points[player]
            total_costs += costs[player]
            chosen_players.append((player, points[player], costs[player], prediction_map[player][1].positions))

    return chosen_players, total_points, total_costs


def get_ordered_lineup(chosen_players):
    """
    Take an unordered list of chosen players from the optimizer, and return a lineup in
    positionally sequenced ordered:
      PG
      SG
      SF
      PF
      C
      G
      F
      UTIL
    """

    # Fill in an ordered lineup, starting with pools of each plaer type
    pgs = [p for p in chosen_players if 'PG' in p[3]]
    sgs = [p for p in chosen_players if 'SG' in p[3]]
    sfs = [p for p in chosen_players if 'SF' in p[3]]
    pfs = [p for p in chosen_players if 'PF' in p[3]]
    cs = [p for p in chosen_players if 'C' in p[3]]
    gs = [p for p in chosen_players if 'PG' in p[3] or 'SG' in p[3]]
    fs = [p for p in chosen_players if 'SF' in p[3] or 'PF' in p[3]]
    utils = [p for p in chosen_players]
    lineup = [None] * 8
    pools = [pgs, sgs, sfs, pfs, cs, gs, fs, utils]

    # Iterate until the lineup is complete
    while not np.all(lineup):

        # First check if any of the positions only has 1 option left, because then the
        # the player has to apply to this position
        one_found = False
        # Check for any position with 1 option left
        for i in range(8):
            if len(pools[i]) == 1:
                one_found = True
                lineup[i] = pools[i][0]
                for pool in pools:
                    if lineup[i] in pool:
                        pool.remove(lineup[i])

        # If there isn't a position with only 1 option left, we have to be smarter
        if not one_found:

            # Check if any of the PG/SG/SF/PF/C positions aren't filled yet, and if not,
            # take the position with the most options remaining and select the player
            # from it who has the least available positions (i.e. just the 1 position)
            if not np.all(lineup[:5]):
                max_i = 0
                max_len = len(pools[0])
                for i in range(1, 5):
                    if len(pools[i]) > max_len:
                        max_i = i
                        max_len = len(pools[i])
                player = min(pools[max_i], key=lambda l: len(l[3]))
                lineup[max_i] = player
                pools[max_i] = []
                for pool in pools:
                    if lineup[max_i] in pool:
                        pool.remove(lineup[max_i])
            # Else if all of the PG/SG/SF/PF/C positions are filled, consider the g and f
            # positions
            elif lineup[5] is None or lineup[6] is None:
                # See if there is a g who is not in the f pool, then we can assume they can
                # be the g
                for g in pools[5]:
                    if g not in pools[6]:
                        lineup[5] = g
                        pools[5] = []
                        pools[7].remove(g)
                        break
                # See if there is a f who is not in the g pool, then we can assume they can
                # be the f
                for f in pools[6]:
                    if f not in pools[5]:
                        lineup[6] = f
                        pools[6] = []
                        pools[7].remove(f)
                        break
                # If there is still no obvious choice above (we would have break-ed), then
                # take one of the g or fs
                if len(pools[5]) > 0:
                    lineup[5] = pools[5][0]
                    pools[5] = []
                    for l in pools[6:]:
                        if lineup[5] in l:
                            l.remove(lineup[5])
                elif len(pools[6]) > 0:
                    lineup[6] = pools[6][0]
                    pools[6] = []
                    for l in [pools[5], pools[7]]:
                        if lineup[6] in l:
                            l.remove(lineup[6])
    return lineup


def optimize_lineup(dk_players, prediction_data, model, model_data_pipeline, players_remove=[], players_add=[]):

    # Get predictions
    predictions = predict_from_model(model, model_data_pipeline, prediction_data)

    # Create mapping of dk player name to tuple of (points prediction, dk_player object)
    prediction_map = {}
    for dk_player in dk_players:
        if dk_player.player_entry is not None and dk_player.player_entry.id in predictions:
            prediction = predictions[dk_player.player_entry.id]
        else:
            prediction = 0.0
        prediction_map[dk_player.dk_player_entry.id] = (prediction, dk_player)

    # Get sorted versions of all players
    sorted_predictions_by_total_points, sorted_predictions_by_value = get_sorted_players(
        prediction_map, print_option="VALUE"
    )

    # Create list of players that aren't being removed
    players = [player for player in list(prediction_map.keys()) if player not in players_remove]

    # Get optimal lineup
    chosen_players, total_points, total_costs = run_lineup_optimizer(
        prediction_map, players, must_include_players=players_add
    )

    # While still debugging problems, print the raw chosen players
    print("Chosen players unordered:")
    for chosen_player in chosen_players:
        print(chosen_player)
    print("Total cost: {}".format(total_costs))
    print("Total predicted points: {}".format(total_points))
    print("---------------------------")

    # Figure out lineup ordering to fulfill positional constraints, and print it
    ordered_lineup = get_ordered_lineup(chosen_players)
    print("Final lineup:")
    for p in ordered_lineup:
        print(p)
    print("---------------------------")
