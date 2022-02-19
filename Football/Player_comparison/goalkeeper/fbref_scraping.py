# I know we can download csv files from FBREF but let's have some fun with beautifulsoup
# the goal is to get the advanced scouting report for 2021-2022
# 
# For this example I will scrap data for Keylor Navas and Gigio Donnarumma

import pandas as pd
from bs4 import BeautifulSoup
import requests

# Function to scrap and transform data to a dataFrame

def scraping_gk_fbref(url):
    navigator = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1)'
    html = requests.get(url, headers={'User-Agent': navigator})
    soup = BeautifulSoup(html.text, 'html.parser')
    
    # We want to scrap data from a table, which look like this :
    # attribute / per90 / percentile
    # we could scrap the whole table and clean the data to get our df
    # it's easier if we scrap each columns rather than the whole table
    
    attribute = soup.find_all("th", {"data-stat" : "statistic"})
    per90 = soup.find_all("td", {"data-stat" : "per90"})
    percentile = soup.find_all("td", {"data-stat" : "percentile"})
    
    # create list that will have the data
    attribute_list = []
    per90_list = []
    percentile_list = []
    
    for item in attribute:
        attribute_list.append(item.text)
    
    for item in per90:
        per90_list.append(item.text)
    
    for item in percentile:
        # need a little bit of transformation
        intermediate = item.text.replace("\xa0", "").replace(" ", "")
        percentile_list.append(intermediate)
    
    # get rid of "Statistic"
    attribute_list = list(filter(lambda a: a !="Statistic", attribute_list))
    
    # empty rows are like this : ("", "", ""), we don't need them
    zipped_list = list(zip(attribute_list, per90_list, percentile_list))
    zipped_list = list(filter(lambda a: a != ("","",""), zipped_list))

    # df creation
    df = pd.DataFrame(zipped_list, columns=["Statistic", "per90", "percentile"])
    # Goal Against, appears 2 times, the second is always at 13
    df.drop(index=13, inplace=True)
    # reset index and drop the column created
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)
    
    return df
  
  
  
# KEYLOR NAVAS 
url = "https://fbref.com/en/players/ecada4fc/scout/11183/Keylor-Navas-Scouting-Report"

df_navas = scraping_gk_fbref(url)

# GIGIO DONNARUMMA
url = "https://fbref.com/en/players/08f5afaa/scout/11183/Gianluigi-Donnarumma-Scouting-Report"

df_donna = scraping_gk_fbref(url)
