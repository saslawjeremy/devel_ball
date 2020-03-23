from mongoengine import (
    Document,
    StringField,
    IntField,
    ListField,
)


class Player(Document):

    """  NBA Player representation. """
    player_id = IntField()
    name = StringField()
    years = ListField()


class GameDate(Document):

    """  GameDate representation. """
    year = StringField()
    date = StringField()
    games = ListField()

class Season(Document):

    """ Season representation. """
    year = StringField()
    first_date = StringField()
    last_date = StringField()
