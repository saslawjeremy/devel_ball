# Various statistical calculators

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




def POSS(FGA, FGM, FTA, OREB_PCT, TO):
    """
    Possessions

    POSS measures the number of possessions played by a player or team
    """
    return (FGA + 0.4*FTA - (1.07*OREB_PCT*FGM) + TO)


def OFF_RATING(PTS, POSS):
    """
    Offensive Rating

    OFF_RATING measures a player's points scored per 100 possessions
    """

################################   TEAM STATS   ################################
def TmAST_PCT(TmAST, TmFGM):
    """
    Team Assist Percentage

    The ratio of field goals assisted by the team
    """
    return TmAST/TmFGM


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
    return (OREB * (TmMIN/5.0))/(MIN * (TmOREB + vsTmOREB))


def DREB_PCT(MIN, DREB, TmMIN, TmDREB, vsTmOREB):
    """
    Defensive Rebound Percentage

    DREB_PCT measures how effective a player is at gaining possession of the
    basketball after a missed field goal or free throw by the other team.

    MIN and DREB apply to player or team
    """
    return (DREB * (TmMIN/5.0))/(MIN * (TmDREB + vsTmDREB))


def AST_TOV(AST, TO):
    """
    Assist to Turnover Ratio

    The number of assists for a player or team compared to the number of
    turnovers they have committed
    """
    return AST/TOV
