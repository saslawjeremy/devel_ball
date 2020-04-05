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
    player_id = StringField()
    name = StringField()
    years = ListField(StringField())


class Team(Document):

    """ Team representation. """
    team_id = StringField()
    name = StringField()


class Season(Document):

    """ Season representation. """
    year = StringField()
    first_date = StringField()
    last_date = StringField()


class Official(Document):

    """ Official representation. """
    official_id = StringField()
    name = StringField()


class GameDate(Document):

    """  GameDate representation. """
    year = StringField()
    date = StringField()
    games = ListField(StringField())


class GameTraditionalStats(EmbeddedDocument):

    """ Traditional stats for a player or team in a single game. """
    MIN = FloatField()  # Must convert to float, e.g. "7:15" -> 7.25
    PTS = IntField()
    FGM = IntField()
    FGA = IntField()
    FG3M = IntField()
    FG3A = IntField()
    FTM = IntField()
    FTA = IntField()
    OREB = IntField()
    DREB = IntField()
    AST = IntField()
    STL = IntField()
    BLK = IntField()
    TO = IntField()
    PF = IntField()
    PLUS_MINUS = IntField()


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


class PlayerGame(Document):

    """ Representation of a player's stats in a single game """
    game_id = StringField()
    player_id = StringField()
    team_id = StringField()
    opposing_team_id = StringField()
    date = StringField()
    home = BooleanField()
    traditional_stats = EmbeddedDocumentField(GameTraditionalStats)
    advanced_stats = EmbeddedDocumentField(GameAdvancedStats)
    usage_stats = EmbeddedDocumentField(GameUsageStats)


class TeamGame(Document):

    """ Representation of a team's stats in a single game """
    game_id = StringField()
    team_id = StringField()
    date = StringField()
    home = BooleanField()
    traditional_stats = EmbeddedDocumentField(GameTraditionalStats)
    advanced_stats = EmbeddedDocumentField(GameAdvancedStats)


class Game(Document):

    """ Game representation. """
    game_id = StringField()
    inactives = ListField(StringField())
    officials = DictField(ReferenceField(Official))  # Key is str(official_id)
    player_games = DictField(ReferenceField(PlayerGame))  # Key is str(player_id)
    team_games = DictField(ReferenceField(TeamGame))  # Key is str(team_id)
