from mongoengine import (
    Document,
    EmbeddedDocument,
    StringField,
    IntField,
    ListField,
    FloatField,
    ReferenceField,
    EmbeddedDocumentField,
    DictField,
    BooleanField
)


class Player(Document):
    """  NBA Player representation. """
    unique_id = StringField(primary_key=True)
    name = StringField()
    years = DictField(ListField())  # Key is season (str), mapping to an ordered
                                    # list of games in that season


class DraftKingsPlayer(Document):
    name = StringField(primary_key=True)
    player = ReferenceField(Player)


class DailyFantasyFuelPlayer(Document):
    name = StringField(primary_key=True)
    dk_player = ReferenceField(DraftKingsPlayer)


class Team(Document):
    """ Team representation. """
    unique_id = StringField(primary_key=True)
    name = StringField()
    years = DictField(ListField())  # Key is season (str), mapping to an ordered
                                    # list of games in that season


class DraftKingsTeam(Document):
    dk_team_id = StringField(primary_key=True)
    team = ReferenceField(Team)


class Official(Document):
    """ Official representation. """
    unique_id = StringField(primary_key=True)
    name = StringField()
    years = DictField(ListField())  # Key is season (str), mapping to an ordered
                                    # list of games in that season


class Season(Document):
    """ Season representation. """
    year = StringField(primary_key=True)
    first_date = StringField()
    last_date = StringField()


class GameDate(Document):
    """  GameDate representation. """
    date = StringField(primary_key=True)
    year = StringField()
    games = ListField(StringField())


class GameTraditionalStats(EmbeddedDocument):
    """ Traditional stats for a player or team in a single game. """
    MIN = FloatField(default=None)  # Must convert to float, e.g. "7:15" -> 7.25
    PTS = FloatField()
    FGM = FloatField()
    FGA = FloatField()
    FG3M = FloatField()
    FG3A = FloatField()
    FTM = FloatField()
    FTA = FloatField()
    OREB = FloatField()
    DREB = FloatField()
    AST = FloatField()
    STL = FloatField()
    BLK = FloatField()
    TO = FloatField()
    PF = FloatField()
    PLUS_MINUS = FloatField()


class GameAdvancedStats(EmbeddedDocument):
    """ Advanced stats for a player or team in a single game. """
    E_OFF_RATING = FloatField()
    OFF_RATING = FloatField()
    E_DEF_RATING = FloatField()
    DEF_RATING = FloatField()
    E_NET_RATING = FloatField()
    NET_RATING = FloatField()
    AST_PCT = FloatField()
    AST_TOV = FloatField()
    AST_RATIO = FloatField()
    OREB_PCT = FloatField()
    DREB_PCT = FloatField()
    REB_PCT = FloatField()
    TM_TOV_PCT = FloatField()
    EFG_PCT = FloatField()
    TS_PCT = FloatField()
    USG_PCT = FloatField()
    E_USG_PCT = FloatField()
    E_PACE = FloatField()
    PACE = FloatField()
    PACE_PER40 = FloatField()
    POSS = FloatField()
    PIE = FloatField()


class GameUsageStats(EmbeddedDocument):
    """ Usage stats for a player in a single game. """
    PCT_FGM = FloatField()
    PCT_FGA = FloatField()
    PCT_FG3M = FloatField()
    PCT_FG3A = FloatField()
    PCT_FTM = FloatField()
    PCT_FTA = FloatField()
    PCT_OREB = FloatField()
    PCT_DREB = FloatField()
    PCT_REB = FloatField()
    PCT_AST = FloatField()
    PCT_TOV = FloatField()
    PCT_STL = FloatField()
    PCT_BLK = FloatField()
    PCT_BLKA = FloatField()
    PCT_PF = FloatField()
    PCT_PFD = FloatField()
    PCT_PTS = FloatField()


class PlayerGame(EmbeddedDocument):
    """ Representation of a player's stats in a single game """
    player_id = StringField()
    home = BooleanField()
    team_id = StringField()
    opposing_team_id = StringField()
    traditional_stats = EmbeddedDocumentField(GameTraditionalStats)
    advanced_stats = EmbeddedDocumentField(GameAdvancedStats)
    usage_stats = EmbeddedDocumentField(GameUsageStats)
    draftkings_points = FloatField()


class TeamGame(EmbeddedDocument):
    """ Representation of a team's stats in a single game """
    home = BooleanField()
    team_id = StringField()
    opposing_team_id = StringField()
    traditional_stats = EmbeddedDocumentField(GameTraditionalStats)
    advanced_stats = EmbeddedDocumentField(GameAdvancedStats)


class Game(Document):
    """ Game representation. """
    game_id = StringField(primary_key=True)
    date = StringField()
    year = StringField()
    inactives = ListField(StringField())
    officials = ListField(StringField())
    player_games = DictField(EmbeddedDocumentField(PlayerGame))  # Key is player_id
    team_games = DictField(EmbeddedDocumentField(TeamGame))  # Key is team_id


class OfficialStatsPerGame(EmbeddedDocument):
    """ Official stats per game on a given day """
    PTS = FloatField()
    FGA = FloatField()
    FTA = FloatField()
    POSS = FloatField()
    PACE = FloatField()
    eFG_PCT = FloatField()
    TS_PCT = FloatField()


class OfficialSeasonDate(EmbeddedDocument):
    """ A given day over the course of a season for a specific official.
        Includes:
            - basic things about the day such as date and game
            - stats going into that game
    """

    game_id = StringField()
    date = StringField()
    season_index = IntField()
    stats_per_game = EmbeddedDocumentField(OfficialStatsPerGame)


class OfficialSeason(Document):
    """ An official's stats over a season """
    official_id = StringField()
    year = StringField()
    season_stats = ListField(EmbeddedDocumentField(OfficialSeasonDate))
    current_stats = EmbeddedDocumentField(OfficialStatsPerGame, default=OfficialStatsPerGame)


class TeamAdvancedStatsPerGame(EmbeddedDocument):
    """ Team advanced stats per game on a given day """
    POSS = FloatField()
    AST_PCT = FloatField()
    PACE = FloatField()
    PIE = FloatField()
    REB_PCT = FloatField()
    OREB_PCT = FloatField()
    DREB_PCT = FloatField()
    AST_TOV = FloatField()
    TO_PCT = FloatField()
    eFG_PCT = FloatField()
    TS_PCT = FloatField()


class TeamStats(EmbeddedDocument):
    """ Various stats pertaining to a player """
    per_game = EmbeddedDocumentField(GameTraditionalStats, default=GameTraditionalStats)
    advanced = EmbeddedDocumentField(TeamAdvancedStatsPerGame, default=TeamAdvancedStatsPerGame)


class TeamSeasonDate(EmbeddedDocument):
    """ A given day over the course of a season for a specific team.
        Includes:
            - basic things about the day such as date, game, opposing team, officials
            - stats going into that game (basic, advanced)
    """

    game_id = StringField()
    date = StringField()
    season_index = IntField()
    home = BooleanField()
    opposing_team_id = StringField()
    officials = ListField(StringField())
    stats = EmbeddedDocumentField(TeamStats)


class TeamSeason(Document):
    """ A team's stats over a season """
    team_id = StringField()
    year = StringField()
    season_stats = ListField(EmbeddedDocumentField(TeamSeasonDate))
    current_stats = EmbeddedDocumentField(TeamStats, default=TeamStats)


class PlayerAdvancedStatsPerGame(EmbeddedDocument):
    """ Player advanced stats per game on a given day """
    POSS = FloatField()
    AST_PCT = FloatField()
    PER = FloatField()
    USG_PCT = FloatField()
    OFF_RTG = FloatField()
    FLOOR_PCT = FloatField()
    DEF_RTG = FloatField()
    GAME_SCORE = FloatField()
    PIE = FloatField()
    REB_PCT = FloatField()
    OREB_PCT = FloatField()
    DREB_PCT = FloatField()
    AST_TOV = FloatField()
    TO_PCT = FloatField()
    eFG_PCT = FloatField()
    TS_PCT = FloatField()


class PlayerRecentStats(EmbeddedDocument):
    """ Player stats for the recent games """
    MIN_RECENT_FIRST = ListField(default=lambda: [None]*10)
    POSS_RECENT_FIRST = ListField(default=lambda: [None]*10)
    USG_PCT_RECENT_FIRST = ListField(default=lambda: [None]*10)
    PTS_RECENT_FIRST = ListField(default=lambda: [None]*10)
    REB_RECENT_FIRST = ListField(default=lambda: [None]*10)
    AST_RECENT_FIRST = ListField(default=lambda: [None]*10)
    STL_RECENT_FIRST = ListField(default=lambda: [None]*10)
    BLK_RECENT_FIRST = ListField(default=lambda: [None]*10)
    TO_RECENT_FIRST = ListField(default=lambda: [None]*10)


class PlayerResults(EmbeddedDocument):
    """ Results for this player in this game, that will try to be predicted. """
    DK_POINTS = FloatField()
    MIN = FloatField()
    POSS = FloatField()
    PTS = IntField()
    REB = IntField()
    AST = IntField()
    FG3M = IntField()
    BLK = IntField()
    STL = IntField()
    TO = IntField()


class PlayerStats(EmbeddedDocument):
    """ Various stats pertaining to a player """
    per_game = EmbeddedDocumentField(GameTraditionalStats, default=GameTraditionalStats)
    per_minute = EmbeddedDocumentField(GameTraditionalStats, default=GameTraditionalStats)
    per_possession = EmbeddedDocumentField(GameTraditionalStats, default=GameTraditionalStats)
    advanced = EmbeddedDocumentField(PlayerAdvancedStatsPerGame, default=PlayerAdvancedStatsPerGame)
    recent = EmbeddedDocumentField(PlayerRecentStats, default=PlayerRecentStats)


class PlayerSeasonDate(EmbeddedDocument):
    """ A given day over the course of a season for a specific player.
        Includes:
            - basic things about the day such as date, game, opposing team, officials
            - traditional stats going into that game (per/game, per/min, per/poss)
            - advanced stats going into that game
    """

    game_id = StringField()
    date = StringField()
    season_index = IntField()
    home = BooleanField()
    team_id = StringField()
    opposing_team_id = StringField()
    officials = ListField(StringField())
    stats = EmbeddedDocumentField(PlayerStats)
    results = EmbeddedDocumentField(PlayerResults)


class PlayerSeason(Document):
    """ A player's stats over a season """
    player_id = StringField()
    year = StringField()
    season_stats = ListField(EmbeddedDocumentField(PlayerSeasonDate))
    current_stats = EmbeddedDocumentField(PlayerStats, default=PlayerStats)
