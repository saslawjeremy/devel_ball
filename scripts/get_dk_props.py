from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
from time import sleep
import pandas
import attr
from mongoengine import connect
from mongoengine import (
    Document,
    StringField,
    FloatField,
    IntField,
    DateTimeField,
)
from datetime import (
    datetime,
    timedelta,
)


class Prop(Document):
    date_created = DateTimeField(default=datetime.utcnow)
    date_updated = DateTimeField(default=datetime.utcnow)
    team1 = StringField()
    team2 = StringField()
    gametime = DateTimeField()
    player = StringField()
    category = StringField()
    over = FloatField()
    over_odds = IntField()
    under = FloatField()
    under_odds = IntField()


@attr.s
class CATEGORY(object):
    name = attr.ib()
    category = attr.ib()
    subcategory = attr.ib()


CATEGORIES = [
    CATEGORY(
        name='Points',
        category='player-points',
        subcategory='points',
    ),
    CATEGORY(
        name='Rebounds',
        category='player-rebounds',
        subcategory='rebounds',
    ),
    CATEGORY(
        name='Assists',
        category='player-assists',
        subcategory='assists',
    ),
    CATEGORY(
        name='Threes',
        category='player-threes',
        subcategory='threes',
    ),
    CATEGORY(
        name='Pts+Reb+Ast',
        category='player-combos',
        subcategory='pts-+-reb-+-ast',
    ),
    # TODO (JS): support later
    # CATEGORY(
    #     name='Double Double',
    #     category='player-combos',
    #     subcategory='double-double',
    # ),
    # CATEGORY(
    #     name='Triple Double',
    #     category='player-combos',
    #     subcategory='triple-double',
    # ),
    CATEGORY(
        name='Pts+Reb',
        category='player-combos',
        subcategory='pts-+-reb',
    ),
    CATEGORY(
        name='Pts+Ast',
        category='player-combos',
        subcategory='pts-+-ast',
    ),
    CATEGORY(
        name='Ast+Reb',
        category='player-combos',
        subcategory='ast-+-reb',
    ),
    CATEGORY(
        name='Blocks',
        category='player-blocks/steals',
        subcategory='blocks-',
    ),
    CATEGORY(
        name='Steals',
        category='player-blocks/steals',
        subcategory='steals-',
    ),
    CATEGORY(
        name='Steals+Blocks',
        category='player-blocks/steals',
        subcategory='steals-+-blocks',
    ),
    CATEGORY(
        name='Turnovers',
        category='player-turnovers',
        subcategory='turnovers',
    ),
]


def get_gametime(todayOrTomorrow, time):
    date = datetime.now() if todayOrTomorrow == 'TODAY' else datetime.now() + timedelta(days=1)
    amOrPm = 'AM' if 'AM' in time else 'PM'
    timeNumber = time.split(amOrPm)[0]
    hour = int(timeNumber.split(':')[0])
    if amOrPm == 'PM':
        hour += 12
    mins = int(timeNumber.split(':')[1])
    gametime = datetime(date.year, date.month, date.day, hour, mins)
    # TODO (JS): figure out timezones
    return gametime


def get_data(driver, url, category):

    # Load page and sleep so it finishes loading
    driver.get(url)
    sleep(5)

    # Get the tables without teams/time
    playerTables = pandas.read_html(driver.page_source)
    # Clean them
    for playerTable in playerTables:
        playerTable.insert(2, "OVER_odds", None)
        playerTable.insert(4, "UNDER_odds", None)
        for _, player in playerTable.iterrows():
            for bet in ('OVER', 'UNDER'):
                # - is represented by chr(8722)
                if chr(8722) in player[bet]:
                    split = player[bet].split(chr(8722))
                    val = float(split[0].split('\xa0')[1])
                    odds = -int(split[1])
                elif '+' in player[bet]:
                    split = player[bet].split('+')
                    val = float(split[0].split('\xa0')[1])
                    odds = int(split[1])
                else:
                    raise Exception("Unexpected state, investigate!")
                player[bet] = val
                player['{}_odds'.format(bet)] = odds

    # Have to be a bit more creative to get the game names / times (if they're today/tomorrow)
    allGames = driver.find_elements(By.CLASS_NAME, 'sportsbook-event-accordion__wrapper')
    if len(playerTables) != len(allGames):
        raise Exception("Something is wrong, mismatching data when scraping!")
    gameDatas = []
    for game in allGames:
        header = game.find_element(By.CLASS_NAME, 'sportsbook-event-accordion__accordion')

        # Parse the string
        splitByAt = header.text.split('\nat\n')
        team1 = ' '.join(splitByAt[0].split())
        part2List = splitByAt[1].split()
        # The game can either be TODAY, TOMORROW, or neither (if live it seems)
        if 'TODAY' in part2List:
            index = part2List.index('TODAY')
            team2 = ' '.join(part2List[:index])
            gametime = get_gametime(part2List[index:][0], part2List[index:][1])
        elif 'TOMORROW' in part2List:
            index = part2List.index('TOMORROW')
            team2 = ' '.join(part2List[:index])
            gametime = get_gametime(part2List[index:][0], part2List[index:][1])
        else:
            team2 = ' '.join(part2List)
            gametime = 'LIVE'

        gameDatas.append([team1, team2, gametime])

    print("PRINTING DATA:   ")
    for gameData, playerTable in zip(gameDatas, playerTables):
        print("{} vs. {}   ({})".format(gameData[0], gameData[1], gameData[2]))
        print(playerTable)
        print("\n")

    for gameData, playerTable in zip(gameDatas, playerTables):
        for _, player in playerTable.iterrows():
            # Don't save off live games
            if gameData[2] == 'LIVE':
                continue
            # See if existing doc for this prop exists
            doc = Prop.objects(
                team1=gameData[0], team2=gameData[1], gametime=gameData[2], category=category, player=player.PLAYER
            ).first()
            if doc is not None:
                doc.over = player.OVER
                doc.over_odds = player.OVER_odds
                doc.under = player.UNDER
                doc.under_odds = player.UNDER_odds
                doc.date_updated = datetime.utcnow()
            else:
                doc = Prop(
                    team1=gameData[0],
                    team2=gameData[1],
                    gametime=gameData[2],
                    player=player.PLAYER,
                    category=category,
                    over=player.OVER,
                    over_odds=player.OVER_odds,
                    under=player.UNDER,
                    under_odds=player.UNDER_odds
                )
            doc.save()


def get_all_data(driver):
    for category in CATEGORIES:
        url = 'https://sportsbook.draftkings.com/leagues/basketball/nba?category={}&subcategory={}'.format(category.category, category.subcategory)
        print("\nGetting data for: {}".format(category.name))
        print("Scraping: {}".format(url))
        get_data(driver, url, category.name)

def main():
    # Connect to the local mongo client and devel_ball database
    mongo_client = connect('DraftKings_Props')

    # Scrape the data
    options = Options()
    options.add_argument("window-size=2000,1200")
    try:
        # Create driver
        driver = webdriver.Chrome(options=options)
        get_all_data(driver)
    finally:
        # Close the driver
        driver.close()

main()
