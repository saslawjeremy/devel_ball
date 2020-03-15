from mongoengine import (
    Document,
    StringField,
    IntField,
    ListField,
)


class Player(Document):

    """  NBA Player representation. """
    id = IntField()
    full_name = StringField()
    first_name = StringField()
    last_name = StringField()

