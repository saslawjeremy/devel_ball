import pulp
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import glob
import os
from time import sleep
import csv
import time

from devel_ball.analysis import predict_from_model
from devel_ball.create_pandas import get_player_from_name
from devel_ball.models import DailyFantasyFuelPlayer


def get_sorted_players(prediction_map):
    # Sort the predictions by total points
    sorted_predictions_by_total_points = dict(
        sorted(prediction_map.items(), key=lambda item: -item[1][0])
    )
    # Sort the predictions by value (points per cost)
    sorted_predictions_by_value = dict(
        sorted(prediction_map.items(), key=lambda item: -item[1][0] / item[1][1].cost)
    )
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
    only_pgs = {
        player: 1 if "PG" in prediction_map[player][1].positions
        and len(prediction_map[player][1].positions) == 1 else 0 for player in players
    }
    only_sgs = {
        player: 1 if "SG" in prediction_map[player][1].positions
        and len(prediction_map[player][1].positions) == 1 else 0 for player in players
    }
    only_sfs = {
        player: 1 if "SF" in prediction_map[player][1].positions
        and len(prediction_map[player][1].positions) == 1 else 0 for player in players
    }
    only_pfs = {
        player: 1 if "PF" in prediction_map[player][1].positions
        and len(prediction_map[player][1].positions) == 1 else 0 for player in players
    }
    only_cs = {
        player: 1 if "C" in prediction_map[player][1].positions
        and len(prediction_map[player][1].positions) == 1 else 0 for player in players
    }

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
    prob += pulp.lpSum([only_pgs[player] * player_vars[player] for player in players]) <= 3, "Max 3 only PGs"
    prob += pulp.lpSum([only_sgs[player] * player_vars[player] for player in players]) <= 3, "Max 3 only SGs"
    prob += pulp.lpSum([only_sfs[player] * player_vars[player] for player in players]) <= 3, "Max 3 only SFs"
    prob += pulp.lpSum([only_pfs[player] * player_vars[player] for player in players]) <= 3, "Max 3 only PFs"
    prob += pulp.lpSum([only_cs[player] * player_vars[player] for player in players]) <= 2, "Max 2 only Cs"
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


def scrape(url, skip_download=False):
    """
    TODO (JS)
    """

    if not skip_download:

        print("Scraping data")

        # Create driver
        options = Options()
        options.add_argument("window-size=2000,1200")  # Open big enough to see download button (it can dissapear)
        driver = webdriver.Chrome(options=options)

        # Load page
        driver.get(url)

        # Download csv
        driver.find_element_by_class_name('hov-underline').click()
        sleep(5)  # Give time for download to finish

        # Close driver
        driver.close()

    # Load the CSV
    filename = sorted(glob.glob('/Users/Jeremy/Downloads/*'), key=os.path.getctime)[::-1][0]
    # Check for partial download
    if len(filename) >= 10 and filename[-10:] == 'crdownload':
        raise Exception("Filename is {}, indicating a partial download".format(filename))
    # Check for latest file being too old (indicates no download happened)
    if not skip_download and time.time() - os.path.getmtime(filename) > 15:
        raise Exception("Latest download is more than 15 seconds old, indicating a download didn't happen.")
    data = []
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                first_name_i = row.index('first_name')
                last_name_i = row.index('last_name')
                injury_status_i = row.index('injury_status')
                projection_i = row.index('ppg_projection')
                team_i = row.index('team')
            else:
                data.append(
                    (row[first_name_i], row[last_name_i], row[team_i], row[injury_status_i], float(row[projection_i]))
                )

    # Ensure there is a DailyFantasyFuelPlayer for each player found
    players_not_found = []
    scraped_map = {}
    for (first_name, last_name, team, injury_status, projection) in data:
        full_name = "{} {}".format(first_name, last_name)
        daily_fantasy_fuel_player = DailyFantasyFuelPlayer.objects(name=full_name).limit(1).first()
        if not daily_fantasy_fuel_player:
            try:
                # Year arg is ignored for dk_player lookup
                dk_player = get_player_from_name(first_name, last_name, year=None, search_dk_players=True)
            except:
                print("Player not found from DailyFantasyFuel: {} on {}".format(full_name, team))
                players_not_found.append(full_name)
                continue
            daily_fantasy_fuel_player = DailyFantasyFuelPlayer(
                name=full_name,
                dk_player=dk_player,
            )
            daily_fantasy_fuel_player.save()
        scraped_map[daily_fantasy_fuel_player.dk_player] = (projection, injury_status)

    # Return mapping of dk_player -> (projection, injury_status)
    return scraped_map


def check_scraped_data(prediction, scraped_prediction, threshold=0.33):
    """
    Currently, if the scraped prediction is less than 1/3 of ours, we'll assume that
    means something is wrong with our prediction, and eliminate that person.
    """
    if np.isclose(scraped_prediction, 0.0):
        return True, "Scraped prediction is 0.0"
    if np.isclose(prediction, 0.0):
        return True, "Prediction is 0.0"
    if scraped_prediction / prediction < threshold:
        return True, "Scraped prediction is {} ({}% of {})".format(
            scraped_prediction, scraped_prediction*100/prediction, prediction,
        )
    return False, ""


def optimize_lineup(
    dk_players,
    prediction_data,
    model,
    model_data_pipeline,
    scrape_url,
    players_remove=[],
    players_add=[],
    lineups=1,
    debug=False,
):

    # For now, only supports up to (9 - players must add) lineups
    max_lineups = 9 - len(players_add)
    if lineups > max_lineups:
        raise Exception(
            "Only supports up to {} lineups when including {} must includes."
            .format(max_lineups, len(player_add))
        )

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
    sorted_predictions_by_total_points, sorted_predictions_by_value = get_sorted_players(prediction_map)

    # Scrape web for predictions to see who to ignore
    # TODO (JS) check date
    scraped_data = scrape(scrape_url)
    for player, (prediction, dk_player) in prediction_map.items():
        # See if player was scraped
        if dk_player.dk_player_entry in scraped_data:
            scraped_prediction = scraped_data[dk_player.dk_player_entry][0]
            dk_player.injury_status = scraped_data[dk_player.dk_player_entry][1]
        else:
            scraped_prediction = None

        # Check if player is ineligible
        if player in players_remove:
            dk_player.ineligible = True
            dk_player.ineligible_reason = "MANUAL"
        elif scraped_prediction is not None:
            ignore, reason = check_scraped_data(prediction, scraped_prediction)
            if ignore:
                dk_player.ineligible = True
                dk_player.ineligible_reason = reason
        else:
            dk_player.ineligible = True
            dk_player.ineligible_reason = "Player isn't in scraped data"

    # Create list of players that aren't being removed
    players = [player for player, (_, dk_player) in prediction_map.items() if not dk_player.ineligible]

    def run_optimization(players):

        # Get optimal lineup
        chosen_players, total_points, total_costs = run_lineup_optimizer(
            prediction_map, players, must_include_players=players_add
        )

        # If debugging, print the raw chosen players
        if debug:
            print("----------------------- Chosen players unordered: -----------------------")
            for chosen_player in chosen_players:
                print(chosen_player)
            print("---------------------------")

        # Figure out lineup ordering to fulfill positional constraints, and print it
        ordered_lineup = get_ordered_lineup(chosen_players)

        return ordered_lineup, total_costs, total_points

    # Store results from all lineups
    results = []
    ordered_lineup, total_costs, total_points = run_optimization(players)
    results.append((ordered_lineup, total_costs, total_points))

    # Create additional linueps, by removing 1 player from the optimal lineup at a time,
    # starting with the least valueable player and moving up
    players_to_remove = [
        p[0] for p in sorted(ordered_lineup, key=lambda item: item[1]/item[2])
        if p[0] not in players_add
    ]
    for i in range(0, lineups-1):
        players = [
            player for player, (_, dk_player) in prediction_map.items()
            if not dk_player.ineligible and player not in players_to_remove[i]
        ]
        results.append(run_optimization(players))

    # Print the players sorted by value
    print("------------------------ Players sorted by value (points per cost): ------------------------ ")
    print("{:30s} {:10s} {:10s} {:10s} {:10s} {:10s}".format(
        "Player", "Injury", "Prediction", "Cost", "Value", "Ineligible\n")
    )
    for player, (prediction, dk_player) in sorted_predictions_by_value.items(): 
        print(
            "{:30s} {:10s} {:10s} {:10s} {:10s} {:10s}".format(
                player,
                dk_player.injury_status if dk_player.injury_status else '',
                str(round(prediction, 3)),
                str(dk_player.cost),
                str(round(prediction/dk_player.cost*1000, 3)),
                dk_player.ineligible_reason if dk_player.ineligible else '',
            )
        )
    # Print the players sorted by total points
    print("\n------------------------ Players sorted by expected points: ------------------------ ")
    print("{:30s} {:10s} {:10s} {:10s} {:10s} {:10s}".format(
        "Player", "Injury", "Prediction", "Cost", "Value", "Ineligible\n")
    )
    for player, (prediction, dk_player) in sorted_predictions_by_total_points.items():
        print(
            "{:30s} {:10s} {:10s} {:10s} {:10s} {:10s}".format(
                player,
                dk_player.injury_status if dk_player.injury_status else '',
                str(round(prediction, 3)),
                str(dk_player.cost),
                str(round(prediction/dk_player.cost*1000, 3)),
                dk_player.ineligible_reason if dk_player.ineligible else '',
            )
        )

    # Print the resultant lineups
    for (ordered_lineup, total_costs, total_points) in results:
        print("\n--------------------------------------- Lineup: --------------------------------------- ")
        for (player, points, cost, positions) in ordered_lineup:
            print(
                "{:30s} {:10s} {:15s} {:15s} {}".format(
                    player,
                    prediction_map[player][1].injury_status,
                    str(points),
                    str(cost),
                    positions,
                )
            )
        print("\nTotal cost: {}".format(total_costs))
        print("Total predicted points: {}".format(total_points))
    print("---------------------------------------------------------------------------------------\n")

    return results

    ### TODO SOMETHING WITH THIS ###
        #try:
        #    print(scraped_data[dk_player.dk_player_entry])
        #except:
        #    print("NOOOOOOOO: {}".format(player))
    #################################
