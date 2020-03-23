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
