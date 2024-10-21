# Download database
import json
import os

import requests

pageStart = 1 # change if e.g. interrupted or want to continue at later stage
totalCount = 15411 # Current database size as of Oct 2024 (without duplicates)
pageSize = 250
i = (pageStart-1) * 250
ignoreFields = ['legalities','tcgplayer','cardmarket','set']
orderedFields = ['id','name','flavorText','types']

if not os.path.exists('pokemonData/'):
    os.mkdir('pokemonData/')

for page in range(pageStart, int(totalCount/pageSize)+1):
    # Get Pokemon Databases (sorted by early generations)
    r = requests.get(url="https://api.pokemontcg.io/v2/cards?q=supertype:Pokémon&orderBy=set.releaseDate,nationalPokedexNumbers.0&page="+str(page))
    for mon in r.json()['data']:
        i += 1
        # Skip if data already exists locally (or special)
        if os.path.exists('pokemonData/'+mon['name']+'.json') \
                or "\'s" in mon['name']\
                or mon['name'].startswith('Unown ')\
                or mon['name'].startswith('Shining ')\
                or mon['name'].startswith('Surfing ')\
                or mon['name'].startswith('Light ')\
                or mon['name'].startswith('Dark ')\
                or mon['name'].startswith('Detective ')\
                or mon['name'].startswith('Armored ')\
                or mon['name'].startswith('Radiant ')\
                or mon['name'].startswith('Special Delivery ')\
                or mon['name'].endswith(']')\
                or mon['name'].endswith(' ◇')\
                or mon['name'].endswith(' V')\
                or mon['name'].endswith(' V-UNION')\
                or mon['name'].endswith(' VSTAR')\
                or mon['name'].endswith(' VMAX')\
                or mon['name'].endswith(' ex')\
                or mon['name'].endswith('-EX')\
                or mon['name'].endswith('-GX')\
                or mon['name'].endswith(' Form')\
                or mon['name'].endswith(' Forme')\
                or mon['name'].endswith(' on the Ball')\
                or mon['name'].endswith(' δ')\
                or mon['name'].endswith(' G')\
                or mon['name'].endswith(' E4')\
                or mon['name'].endswith(' GL')\
                or mon['name'].endswith(' FB')\
                or mon['name'].endswith(' C')\
                or mon['name'].endswith(' BREAK')\
                or mon['name'].endswith(' LEGEND')\
                or mon['name'].endswith('LV.X')\
                or mon['name'].endswith(' ★'):
            print("p"+str(page), i, "/", totalCount, str(int(100*i/totalCount))+"%", "SKIP  ", mon['name']+'.json')
            continue

        # only keep fields of interest
        for field in ignoreFields:
            if field in mon:
                del(mon[field])

        # reorder fields (for easier generation later)
        orderedDict = {}
        for field in orderedFields:
            if field in mon:
                orderedDict[field] = mon[field]
        for field in mon.keys():
            if not field in orderedDict:
                orderedDict[field] = mon[field]

        # write out every pokemon
        try:
            with open('pokemonData/'+mon['name']+'.json', 'w+') as f:
                json.dump(orderedDict, f)
                print("p"+str(page), i, "/", totalCount, str(int(100*i/totalCount))+"%", "CREATE", mon['name']+'.json')
        except:
            print("p"+str(page), i, "/", totalCount, str(int(100*i/totalCount))+"%", "ERROR ", mon['name']+'.json')