from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium import webdriver
from time import sleep
import pandas

def get_data(driver, url):

    print("Scraping data: {}".format(url))

    # Load page and sleep so it finishes loading
    driver.get(url)
    sleep(5)

    # Get the tables without teams/time
    playerTables = pandas.read_html(driver.page_source)
    # Clean them
    for playerTable in playerTables:
        playerTable.insert(2, "OVER odds", None)
        playerTable.insert(4, "UNDER odds", None)
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
                player['{} odds'.format(bet)] = odds

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
            time = ' '.join(part2List[index:])
        elif 'TOMORROW' in part2List:
            index = part2List.index('TOMORROW')
            team2 = ' '.join(part2List[:index])
            time = ' '.join(part2List[index:])
        else:
            team2 = ' '.join(part2List)
            time = None

        gameDatas.append([team1, team2, time])

    print("PRINTING DATA:   ")
    for gameData, playerTable in zip(gameDatas, playerTables):
        print("{} vs. {}   ({})".format(gameData[0], gameData[1], gameData[2] if gameData[2] else "LIVE"))
        print(playerTable)
        print("\n")

def get_all_data(driver):

    url = 'https://sportsbook.draftkings.com/leagues/basketball/nba?category=player-points'
    datatype = 'POINTS'
    print("\nGetting data for: {}".format(datatype))
    get_data(driver, url)

def main():
    options = Options()
    options.add_argument("window-size=2000,1200")  # Open big enough to see download button (it can dissapear)
    try:
        # Create driver
        driver = webdriver.Chrome(options=options)
        get_all_data(driver)
    finally:
        # Close the driver
        driver.close()

main()
