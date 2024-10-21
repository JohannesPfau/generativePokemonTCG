import json
import string
import random

import requests

# Parameters
model = "Llama 3 8B Instruct"
pokeName = "Ori"
pokeDescription = "A small feline predator that scratches everything with his claws, but can purr to calm friends."
pokeType = "Dark" # One string of the typeIDs
author = "Johannes Pfau" # replace with your name
checkModels = True

# 2nd try
# pokeName = "Atrona"
# pokeDescription = "A female fire elemental made of flames and steel."
# pokeType = "Fire"

if checkModels:
    # Trying to call local gpt4all server
    r = requests.get('http://localhost:4891/v1/models')
    j = r.json()
    print("Local GPT4All server connection successful!")
    print("Available models:")
    selected = False
    for m in j['data']:
        if m['id'] == model:
            print(m['id'], "[SELECTED]")
            selected = True
        else:
            print(m['id'])

    if not selected:
        print("No model selected, or model not available: ", model)

pokeDict = {
    "id": "aicg2024-1",
    "name": pokeName,
    "flavorText": pokeDescription,
    "types": [pokeType],
    "supertype": "Pok\u00e9mon",
    "subtypes": ["Basic"]
}

payload = {
  "model": model,
  "max_tokens": 4096,
  "temperature": 2,
  "messages": [{"role":"user", "content": str(pokeDict).replace("\'","\"")}]
}
with open(pokeName + '_1_LLMprompt.json', 'w+') as f:
    json.dump(payload, f)
r = requests.post('http://localhost:4891/v1/chat/completions', json=payload)
j = r.json()
print("Raw output:")
print(j)
with open(pokeName + '_2_LLMresponse.json', 'w+') as f:
    json.dump(j, f)
answer = json.loads(j["choices"][0]["message"]["content"])
print("Generated Pokémon card mechanics based on name, type and description:")
print(answer)
print("The model used the following reference files from LocalDocs:")
for ref in j["choices"][0]["references"]:
    print(ref["file"])

# Unite pre and post dict
finalPokeDict = {}
for key, val in pokeDict.items():
    finalPokeDict[key] = val
for key, val in answer.items():
    finalPokeDict[key] = val
print("Unified JSON:")
print(finalPokeDict)
with open(pokeName + '_3_LLMunion.json', 'w+') as f:
    json.dump(finalPokeDict, f)

# TRANSFORM
def typeToID(t):
    typeIDs = {
        "Grass": 1,
        "Fire": 2,
        "Water": 3,
        "Lightning": 4,
        "Psychic": 5,
        "Fighting": 6,
        "Dark": 7,
        "Darkness": 7,
        "Metal": 8,
        "Fairy": 9,
        "Dragon": 10,
        "Colorless": 11
    }
    for type, typeID in typeIDs.items():
        if t == type:
            return typeID
    print("WARNING: type ID could not be found:", t)
    return -1

with open('pokeTemplate.json', 'r') as f:
    pokeTemplate = json.load(f)
pokeTemplate["name"] = pokeName
pokeTemplate["dexEntry"] = pokeDescription

if "types" in finalPokeDict and isinstance(finalPokeDict["types"], list):
    pokeTemplate["typeId"] = typeToID(finalPokeDict["types"][0])
    if pokeTemplate["typeId"] == -1:
        pokeTemplate["typeId"] = 11 # Default: Colorless

if "hitpoints" in finalPokeDict and isinstance(finalPokeDict["hitpoints"], int):
    pokeTemplate["hitpoints"] = finalPokeDict["hp"]
if "retreatCost" in finalPokeDict and isinstance(finalPokeDict["retreatCost"], int):
    pokeTemplate["retreatCost"] = finalPokeDict["retreatCost"]
pokeTemplate["illustrator"] = author

if "resistances" in finalPokeDict and isinstance(finalPokeDict["resistances"], list) and len(finalPokeDict["resistances"]) > 0:
    id = typeToID(finalPokeDict["resistances"][0])
    if id != -1:
        pokeTemplate["resistanceTypeId"] = id
if "weaknesses" in finalPokeDict and isinstance(finalPokeDict["weaknesses"], list) and len(finalPokeDict["weaknesses"]) > 0:
    id = typeToID(finalPokeDict["weaknesses"][0])
    if id != -1:
        pokeTemplate["weaknessTypeId"] = id

pokeTemplate["moves"] = []
if "abilities" in finalPokeDict and isinstance(finalPokeDict["abilities"], list):
    for ability in finalPokeDict["abilities"]:
        move = {
            "id": ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) + "_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)),
            "type": "default",
            "isAbility": True
        }
        if isinstance(ability, str):
            move["name"] = ability
            move["description"] = ""
        else:
            if not isinstance(ability, dict) or not "name" in ability or ability["name"] == "":
                continue
            move["name"] = ability["name"]
            move["description"] = ability["text"]
        pokeTemplate["moves"].append(move)
if "attacks" in finalPokeDict and isinstance(finalPokeDict["attacks"], list):
    for attack in finalPokeDict["attacks"]:
        if not isinstance(attack, dict) or not "name" in attack or attack["name"] == "":
            continue
        move = {
            "id": ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)) + "_" + ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10)),
            "type": "default",
            "isAbility": False,
            "name": attack["name"],
            "description": attack["text"],
            "damageAmount": attack["damage"],
            "energyCost": [{
            "amount": attack["convertedEnergyCost"],
            "typeId": 11 # Default: Colorless
        }]}
        if "cost" in attack and len(attack["cost"]) > 0:
            id = typeToID(attack["cost"][0])
            if id != -1:
                move["energyCost"][0]["typeId"] = id
        pokeTemplate["moves"].append(move)

print("Transformed format:\n")
print(json.dumps(pokeTemplate))
with open(pokeName + '_4_PCMInput.json', 'w+') as f:
    json.dump(pokeTemplate, f)
print("\nCopy and Import into: https://pokecardmaker.net/creator")