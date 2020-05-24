# Various statistical calculators

from collections import namedtuple


###############################   PLAYER STATS   ###############################
def AST_PCT(AST, MIN, FGM, TmMIN, TmFGM):
    """
    Assist Percentage

    Estimate of the percentage of teammate field goals a player assisted while
    he was on the floor
    """
    return AST / (( (MIN/(TmMIN/5.0)) * TmFGM) - FGM)


def PER(FGM, FGA, STL, FG3M, FTM, FTA, BLK, OREB, AST, DREB, PF, TO, MIN):
    """
    Player Efficiency Rating

    Overall rating of a player's per-minute stastical production.
    League average is 15.00 every season
    """
    return (FGM*85.910
            + STL*53.897
            + FG3M*51.757
            + FTM*46.845
            + BLK*39.190
            + OREB*39.190
            + AST*34.677
            + DREB*14.707
            - PF*17.174
            - (FTA-FTM)*20.091
            - (FGA-FGM)*39.190
            - TO*53.897)/MIN


def USG_PCT(FGA, FTA, TO, MIN, TmMIN, TmFGA, TmFTA, TmTO):
    """
    Usage Percentage

    Estimate of percentage of team plays used by a player when he was on floor
    """
    return (((FGA + 0.44*FTA + TO) * TmMIN/5) /
             (MIN * (TmFGA + 0.44*TmFTA + TmTO)))


def _advanced_helper(MIN, PTS, FGM, FGA, FTM, FTA, OREB, AST, TO,
                     TmMIN, TmPTS, TmFGM, TmFGA, TmFTM, TmFTA, TmOREB,
                     TmAST, TmTO, TmOREB_PCT):
    """
    Helper function to get internal statistics useful for multiple stats
    """

    # Assist adjustment factor is the quantity of field goals assisted by
    # teammates while the player of interest is in the game plus the expected
    # rate of assists per field goal made for the player of interest
    qAST = (((MIN / (TmMIN/5.0)) * (1.14 * ((TmAST - AST) / TmFGM))) +
            ((((TmAST / TmMIN) * MIN * 5.0 - AST) / ((TmFGM / TmMIN) * MIN * 5.0 - FGM))
               * (1.0 - (MIN / (TmMIN / 5.0)))))

    # Possessions ending due to field goals
    TmScoringPoss = TmFGM + (1.0 - (1.0 - (TmFTM/TmFTA))**2.0) * TmFTA * 0.4

    # Number of scoring possessions divded by number of possessions played
    TmPlayPct = TmScoringPoss / (TmFGA + TmFTA*0.4 + TmTO)

    # Percentage of offensive rebounds obtained from the total number of
    # possible rebounds of an offensive possession
    TmOREB_Weight = (
        ((1.0 - TmOREB_PCT) * TmPlayPct) /
        ((1.0 - TmOREB_PCT) * TmPlayPct + TmOREB_PCT*(1.0-TmPlayPct)))

    # Identify the points produced by a player with respect to field goals made
    FG_Part = FGM * (1.0 - 0.5 * ((PTS - FTM) / (2.0 * FGA)) * qAST)

    # Identify the assists produced by a player
    AST_Part = 0.5 * (((TmPTS - TmFTM) - (PTS - FTM)) / (2.0 * (TmFGA - FGA))) * AST

    # Free throw part
    FT_Part = (1.0 - (1.0 - (FTM/FTA))**2) * 0.4 * FTA

    # Offensive rebounding part
    OREB_Part = OREB * TmOREB_Weight * TmPlayPct

    # Scoring Possessions
    ScPoss = ((FG_Part + AST_Part + FT_Part) * (1.0 - (TmOREB/TmScoringPoss)
              * TmOREB_Weight * TmPlayPct) + OREB_Part)

    # Missed FG and FT Possessions
    missed_FG_POSS = (FGA - FGM) * (1.0 - 1.07*TmOREB_PCT)
    missed_FT_POSS = ((1.0 - (FTM/FTA))**2.0) * 0.4 * FTA

    # Total Possessions
    TotPoss = ScPoss + missed_FG_POSS + missed_FT_POSS + TO

    # Return named tuple with relevant stats
    helper_stats = namedtuple(
        'HelperStats',
        ['qAST', 'TmScoringPoss', 'TmPlayPct', 'TmOREB_Weight', 'ScPoss',
            'TotPoss']
    )
    helper_stats.qAST = qAST
    helper_stats.TmScoringPoss = TmScoringPoss
    helper_stats.TmPlayPct = TmPlayPct
    helper_stats.TmOREB_Weight = TmOREB_Weight
    helper_stats.ScPoss = ScPoss
    helper_stats.TotPoss = TotPoss
    return helper_stats


def OFF_RTG(MIN, PTS, FGM, FGA, FG3M, FTM, FTA, OREB, AST, TO,
            TmMIN, TmPTS, TmFGM, TmFGA, TmFG3M, TmFTM, TmFTA, TmOREB,
            TmAST, TmTO, TmOREB_PCT):
    """
    Offensive Rating

    Number of points produced by a player per hundred total possessions.
    In other words, how many points is a player likely to generate when he tries.

    The key building blocks for offensive rating are:
    Individual Total Possessions and Individual Points Produced

    The formula for Total Possessions is broken down into 4 components:
    Scoring Possessions, Missed FG Possessions, Missed FT Possessions, Turnovers
    """

    helper_stats =  _advanced_helper(
        MIN, PTS, FGM, FGA, FTM, FTA, OREB, AST, TO,
        TmMIN, TmPTS, TmFGM, TmFGA, TmFTM, TmFTA, TmOREB, TmAST, TmTO, TmOREB_PCT)
    qAST = helper_stats.qAST
    TmScoringPoss = helper_stats.TmScoringPoss
    TmPlayPct = helper_stats.TmPlayPct
    TmOREB_Weight = helper_stats.TmOREB_Weight
    TotPoss = helper_stats.TotPoss

    # Points produced off field goals
    PProd_FG_Part = 2.0 * (FGM + 0.5*FG3M) * (1.0 - 0.5 *((PTS-FTM) / (2*FGA)) * qAST)

    # Points produced off assists
    PProd_AST_Part = (2.0 * ((TmFGM - FGM + 0.5*(TmFG3M - FG3M)) / (TmFGM - FGM))
                      * 0.5 * (((TmPTS - TmFTM) - (PTS - FTM)) / (2.0 * (TmFGA - FGA))) * AST)

    # Points produced off offensive rebounds
    PProd_OREB_Part = (OREB * TmOREB_Weight * TmPlayPct *
                      (TmPTS / (TmFGM + (1.0 - (1.0 - (TmFTM/TmFTA))**2) * 0.4 * TmFTA)))

    # Individual points produced
    PProd = ((PProd_FG_Part + PProd_AST_Part + FTM)
                 * (1.0 - (TmOREB / TmScoringPoss) * TmOREB_Weight * TmPlayPct)
             + PProd_OREB_Part)

    # Calculate offensive rating
    ORtg = 100.0 * (PProd / TotPoss)
    return ORtg


def FLOOR_PCT(MIN, PTS, FGM, FGA, FTM, FTA, OREB, AST, TO,
              TmMIN, TmPTS, TmFGM, TmFGA, TmFTM, TmFTA, TmOREB,
              TmAST, TmTO, TmOREB_PCT):
    """
    Floor Percentage

    What percentage of the time that a player wants to score does he actually score
    """

    helper_stats =  _advanced_helper(
        MIN, PTS, FGM, FGA, FTM, FTA, OREB, AST, TO,
        TmMIN, TmPTS, TmFGM, TmFGA, TmFTM, TmFTA, TmOREB, TmAST, TmTO, TmOREB_PCT)
    ScPoss = helper_stats.ScPoss
    TotPoss = helper_stats.TotPoss

    FLOOR_PCT = ScPoss/TotPoss
    return FLOOR_PCT


def DEF_RTG(MIN, STL, BLK, DREB, PF,
            TmMIN, TmDREB, TmBLK, TmSTL, TmPF, TmPOSS,
            vsTmMIN, vsTmPTS, vsTmOREB, vsTmFGM, vsTmFGA, vsTmFTM, vsTmFTA, vsTmTO):
    """
    Defensive Rating

    Estimates how many points the player allowed per 100 possessions he individually
    faced while on the court.
    """

    DOR_PCT = vsTmOREB / (vsTmOREB + TmDREB)
    DFG_PCT = vsTmFGM / vsTmFGA

    FMwt = (DFG_PCT * (1.0 - DOR_PCT)) / (DFG_PCT * (1.0 - DOR_PCT) + (1.0 - DFG_PCT) * DOR_PCT)

    Stops1 = STL + BLK*FMwt * (1.0 - 1.07*DOR_PCT) + DREB*(1.0 - FMwt)

    Stops2 = ((((vsTmFGA - vsTmFGM - TmBLK) / TmMIN) * FMwt * (1.0 - 1.07*DOR_PCT)
               + ((vsTmTO - TmSTL) / TmMIN)) * MIN + (PF/TmPF) * 0.4 * vsTmFTA *
               (1.0 - (vsTmFTM/vsTmFTA))**2)

    Stops = Stops1 + Stops2

    Stop_PCT = (Stops * vsTmMIN) / (TmPOSS*MIN)

    TmDEF_RTG = 100 * (vsTmPTS/TmPOSS)
    D_Pts_per_ScPoss = vsTmPTS / (vsTmFGM + (1.0 - (1.0 - (vsTmFTM/vsTmFTA))**2) * vsTmFTA*0.4)

    DEF_RTG = TmDEF_RTG + 0.2*(100.0*D_Pts_per_ScPoss * (1.0 - Stop_PCT) - TmDEF_RTG)
    return DEF_RTG


def GAME_SCORE(PTS, FGM, FGA, FTM, FTA, OREB, DREB, STL, AST, BLK, PF, TO):
    """
    Game Score

    Give a rough measure of a player's productivity for a single game.
    The scale is similar to that of points scored (40 is outstanding, 10 average)
    """
    return (PTS + 0.4*FGM - 0.7*FGA -0.4*(FTM-FTA) + 0.7*OREB + 0.3*DREB +
            STL + 0.7*AST + 0.7*BLK - 0.4*PF - TO)


################################   TEAM STATS   ################################
def TmAST_PCT(TmAST, TmFGM):
    """
    Team Assist Percentage

    The ratio of field goals assisted by the team
    """
    return TmAST/TmFGM


def PACE(TmMIN, TmPOSS, vsTmPOSS):
    """
    Pace Factor

    An estimate of the number of possessions per 48 minutes by a team
    """
    return 48*((TmPOSS + vsTmPOSS) / (2*(TmMIN/5.0)))


#############################   PLAYER+TEAM STATS   ############################

def PIE(PTS, FGM, FTM, FGA, FTA, DREB, OREB, AST, STL, BLK, PF, TO,
        GmPTS, GmFGM, GmFTM, GmFGA, GmFTA, GmDREB, GmOREB, GmAST, GmSTL, GmBLK,
        GmPF, GmTO):
    """
    Player Impact Estimate

    PIE measures a player or team's overall statistical contribution against the
    total statistics in games they play in.

    First row of params are player or team's stats, and the 2nd/3rd rows are
    game stats
    """
    return ((PTS + FGM + FTM - FGA - FTA + DREB + (0.5*OREB) + AST + STL
                 + (0.5*BLK) - PF - TO)
            /
            (GmPTS + GmFGM + GmFTM - GmFGA - GmFTA + GmDREB + (0.5*GmOREB)
             + GmAST + GmSTL + (0.5*GmBLK) - GmPF - GmTO)
    )


def REB_PCT(MIN, REB, TmMIN, TmREB, vsTmREB):
    """
    Rebound Percentage

    REB_PCT measures how effective a player or team is at gaining possession of
    the basketball after a missed field goal or free throw.

    MIN and REB apply to player or team
    """
    return (REB * (TmMIN/5.0))/(MIN * (TmREB + vsTmREB))


def OREB_PCT(MIN, OREB, TmMIN, TmOREB, vsTmDREB):
    """
    Offensive Rebound Percentage

    OREB_PCT measures how effective a player is at gaining possession of the
    basketball after a missed field goal or free throw by their team.

    MIN and OREB apply to player or team
    """
    return (OREB * (TmMIN/5.0))/(MIN * (TmOREB + vsTmDREB))


def DREB_PCT(MIN, DREB, TmMIN, TmDREB, vsTmOREB):
    """
    Defensive Rebound Percentage

    DREB_PCT measures how effective a player is at gaining possession of the
    basketball after a missed field goal or free throw by the other team.

    MIN and DREB apply to player or team
    """
    return (DREB * (TmMIN/5.0))/(MIN * (TmDREB + vsTmOREB))


def AST_TOV(AST, TO):
    """
    Assist to Turnover Ratio

    The number of assists for a player or team compared to the number of
    turnovers they have committed
    """
    return AST/TOV


def TO_PCT(TO, FGA, FTA):
    """
    Turnover percentage

    Percentage of plays that end in a player or team's turnover
    """
    return TO/(FGA + 0.44*FTA + TO)


def eFG_PCT(FGM, FG3M, FGA):
    """
    Effective Field Goal Percentage

    Measures field goal percentage taking into account 3 pointers worth more
    """
    return (FGM + 0.5*FG3M)/FGA


def TS_PCT(PTS, FGA, FTA):
    """
    True Shooting Percentage

    Shooting percentage taking into account FG, 3FG, and FT
    """
    return PTS/(2.0*FGA + 0.88*FTA)


