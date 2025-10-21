import json
import os
import string
import random
import urllib
import uuid
import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import requests
import websocket
from PIL import Image

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
    return 11

def listAvailableModels(LMStudio_server_address):
    r = requests.get(LMStudio_server_address + '/v1/models')
    j = r.json()
    print("Available models:")
    for model in j['data']:
        print(model['id'])

def promptLMStudio(pokeName, pokeDict, model, systemPrompt, LMStudio_server_address, use_RAG=True, num_RAG_samples=5, temperature=1):
    if use_RAG:
        RAGresults = RAG(pokeDict, num_RAG_samples)
        if len(RAGresults) > 0:
            systemPrompt += "\n\nTake these similar Pokémon cards as reference. Make the response similar but unique and novel:"
            for result in RAGresults:
                systemPrompt += "\n\n" + str(result).replace("\'","\"")

    pokemonSchema = json.load(open('pokemon_tcg_schema.json'))

    payload = {
        "model": model,
        "max_tokens": 10000,
        "seed": random.randint(0, 123456),
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": systemPrompt},
            {"role":"user", "content": str(pokeDict).replace("\'","\"")},
            {"role": "system", "content": "{\n  \"id\": \""+ pokeDict["id"] +"\",\n  \"name\": \""+ pokeDict["name"] +"\",\n  \"flavorText\": \""+ pokeDict["flavorText"] +"\",\n  \"types\": [\""+ pokeDict["types"][0] +"\"],\n  \"supertype\": \"Pokémon\",\n  \"subtypes\": [\"Basic\"],\n"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pokemon_trading_card",
                "strict": True,
                "schema": pokemonSchema
            }
        }
    }
    # Save the prompt to a file
    with open(os.path.join("output", pokeName + '_1_LLMprompt.json'), 'w+') as f:
        json.dump(payload, f)

    print("Sending prompt to LMStudio server at " + LMStudio_server_address + '/v1/chat/completions')
    r = requests.post(LMStudio_server_address + '/v1/chat/completions', json=payload)
    j = r.json()
    print("Raw output:")
    print(j)
    with open(os.path.join("output", pokeName + '_2_LLMresponse.json'), 'w+') as f:
        json.dump(j, f)
    print("Mechanics Reponse:")
    print(j["choices"][0]["message"]["content"])
    answer = json.loads(j["choices"][0]["message"]["content"])
    print("Generated Pokémon card mechanics based on name, type and description:")
    print(answer)
    # print("The model used the following reference files from LocalDocs:")
    # for ref in j["choices"][0]["references"]:
    #     print(ref["file"])

    # Unite pre and post dict
    finalPokeDict = {}
    for key, val in pokeDict.items():
        finalPokeDict[key] = val
    for key, val in answer.items():
        finalPokeDict[key] = val
    print("Unified JSON:")
    print(finalPokeDict)
    with open(os.path.join("output", pokeName + '_3_LLMunion.json'), 'w+') as f:
        json.dump(finalPokeDict, f)


    with open('pokeTemplate.json', 'r') as f:
        pokeTemplate = json.load(f)
    pokeTemplate["name"] = pokeName
    pokeTemplate["dexEntry"] = pokeDict["flavorText"]

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
    with open(os.path.join("output", pokeName + '_4_PCMInput.json'), 'w+') as f:
        json.dump(pokeTemplate, f)
    print("\nNow Import JSON " + pokeName + "_4_PCMInput.json into: https://pokecardmaker.net/creator")

def open_websocket_connection(server_address):
    client_id = str(uuid.uuid4())
    # Remove http:// prefix and trailing slash if present
    clean_address = server_address.replace('http://', '').replace('https://', '').rstrip('/')
    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(clean_address, client_id))
    return ws, clean_address, client_id

def queue_prompt(prompt, client_id, server_address):
    p = {"prompt": prompt, "client_id": client_id}
    headers = {'Content-Type': 'application/json'}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
    return json.loads(urllib.request.urlopen(req).read())

def track_progress(prompt, ws, prompt_id):
    node_ids = list(prompt.keys())
    finished_nodes = []

    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'progress':
                data = message['data']
                current_step = data['value']
                print('In K-Sampler -> Step: ', current_step, ' of: ', data['max'])
            if message['type'] == 'execution_cached':
                data = message['data']
                for itm in data['nodes']:
                    if itm not in finished_nodes:
                        finished_nodes.append(itm)
                        print('Progress: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] not in finished_nodes:
                    finished_nodes.append(data['node'])
                    print('Progress: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')

                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break  # Execution is done
        else:
            continue
    return

def get_history(prompt_id, server_address):
    """Get the execution history for a completed prompt"""
    try:
        req = urllib.request.Request("http://{}/history/{}".format(server_address, prompt_id))
        response = urllib.request.urlopen(req)
        return json.loads(response.read())
    except Exception as e:
        print("Error getting history:", str(e))
        return None

def download_image(image_url, server_address, output_path):
    """Download image from ComfyUI server to local output folder"""
    try:
        # Construct full URL
        full_url = "http://{}/view?filename={}".format(server_address, image_url)
        print("Downloading image from:", full_url)
        
        # Download the image
        req = urllib.request.Request(full_url)
        response = urllib.request.urlopen(req)
        
        # Save to output folder
        with open(output_path, 'wb') as f:
            f.write(response.read())
        
        print("Image saved to:", output_path)
        return True
    except Exception as e:
        print("Error downloading image:", str(e))
        return False

def promptComfyUI(diffusion_prompt, workflow_template, steps, cfg, denoise, lora_strength_niji, lora_strength_pokemon, comfyUI_server_address, randomize_seed=True):
    print("Prompting ComfyUI with:", diffusion_prompt)
    
    # Load the ComfyUI workflow template
    with open(workflow_template, 'r') as f:
        workflow = json.load(f)
    
    # Replace PROMPT_PLACEHOLDER with the diffusion_prompt
    workflow["16"]["inputs"]["text"] = diffusion_prompt
    workflow["7"]["inputs"]["steps"] = steps
    workflow["7"]["inputs"]["denoise"] = denoise
    workflow["9"]["inputs"]["guidance"] = cfg
    workflow["15"]["inputs"]["strength_model"] = lora_strength_niji
    workflow["17"]["inputs"]["strength_model"] = lora_strength_pokemon
    if randomize_seed:
        workflow["6"]["inputs"]["noise_seed"] = random.randint(0, 123456) # Random seed for the noise

    # Save the modified workflow
    with open(os.path.join("output", pokeName + '_5_ComfyUIworkflow.json'), 'w+') as f:
        json.dump(workflow, f, indent=2)
    
    print("ComfyUI workflow saved to output/" + pokeName + "_5_ComfyUIworkflow.json")
    
    # Send request to ComfyUI server
    try:
        print("Sending workflow to ComfyUI server at:", comfyUI_server_address)
        ws, server_address, client_id = open_websocket_connection(comfyUI_server_address)
        print("WebSocket connected successfully to:", server_address)
        prompt_id = queue_prompt(workflow, client_id, server_address)['prompt_id']
        print("Workflow queued with ID:", prompt_id)
        track_progress(workflow, ws, prompt_id)
        print("Image generation completed!")
        
        # Retrieve and save the generated image
        print("Retrieving generated image...")
        history = get_history(prompt_id, server_address)
        if history and prompt_id in history:
            prompt_data = history[prompt_id]
            if 'outputs' in prompt_data:
                # Look for SaveImage node output (usually node "10" in our workflow)
                for node_id, node_output in prompt_data['outputs'].items():
                    if 'images' in node_output:
                        for image_info in node_output['images']:
                            filename = image_info['filename']
                            # Create output path
                            output_path = os.path.join("output", pokeName + "_6_ComfyUIoutput.png")
                            # Download the image
                            if download_image(filename, server_address, output_path):
                                print(f"Successfully saved generated image: {output_path}")
                                createFramedPicture(output_path)
                            else:
                                print("Failed to download generated image")
                        break
        else:
            print("Could not retrieve image from ComfyUI server")
            
    except Exception as e:
        print("Error connecting to ComfyUI server:", str(e))
        print("Make sure the ComfyUI server is running and accessible at:", comfyUI_server_address)
    finally:
        if 'ws' in locals():
            ws.close()

def createFramedPicture(imgpath):
    """
    Create a transparent background image and paste the input image at specified offset.
    Save the result with 'ComfyUIoutput' replaced with 'PokeCardMakerInput' in filename.
    """
    try:
        # Create a transparent background image (1484x2074 pixels)
        background = Image.new('RGBA', (1484, 2074), color=(255, 255, 255, 0))
        
        # Load the input image
        input_image = Image.open(imgpath)
        
        # Define the offset position
        x_offset = 115
        y_offset = 201
        
        # Paste the input image onto the white background at the specified offset
        background.paste(input_image, (x_offset, y_offset))
        
        # Create output filename by replacing 'ComfyUIoutput' with 'PokeCardMakerInput'
        output_filename = imgpath.replace('6_ComfyUIoutput.png', '7_PokeCardMakerInput.png')
        
        # Save the new image
        background.save(output_filename)
        
        print(f"Framed picture saved as: {output_filename}")
        return output_filename
        
    except Exception as e:
        print(f"Error creating framed picture: {str(e)}")
        return None

def RAG(pokeTCGDict, num_results=5):
    """
    Perform retrieval-augmented generation by scanning pokemonData folder.
    Calculate cosine similarity between input pokeTCGDict and all JSON documents.
    Return the num_results closest matches.
    """
    try:
        # Get all JSON files in pokemonData folder
        json_files = glob.glob("pokemonData/*.json")
        
        if not json_files:
            print("No JSON files found in pokemonData folder")
            return []
        
        # Load all Pokemon data
        pokemon_docs = []
        pokemon_names = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    pokemon_data = json.load(f)
                    pokemon_docs.append(pokemon_data)
                    pokemon_names.append(os.path.basename(json_file).replace('.json', ''))
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")
                continue
        
        if not pokemon_docs:
            print("No valid Pokemon data loaded")
            return []
        
        # Create text representations for similarity calculation
        def create_text_representation(pokemon_dict):
            """Create a text representation of Pokemon data for similarity calculation"""
            text_parts = []
            
            # Add basic info
            if 'name' in pokemon_dict:
                text_parts.append(pokemon_dict['name'])
            if 'types' in pokemon_dict and pokemon_dict['types']:
                text_parts.extend(pokemon_dict['types'])
            if 'flavorText' in pokemon_dict:
                text_parts.append(pokemon_dict['flavorText'])
            
            return ' '.join(text_parts)
        
        # Create text representations
        query_text = create_text_representation(pokeTCGDict)
        doc_texts = [create_text_representation(doc) for doc in pokemon_docs]
        
        # Add query to the corpus for vectorization
        all_texts = [query_text] + doc_texts
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', lowercase=True, ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarities
        query_vector = tfidf_matrix[0:1]  # First vector is the query
        doc_vectors = tfidf_matrix[1:]   # Rest are the documents
        
        similarities = cosine_similarity(query_vector, doc_vectors).flatten()
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1][:num_results]
        
        # Prepare results
        results = []
        for i, idx in enumerate(top_indices):
            if similarities[idx] > 0:  # Only include results with some similarity
                result = {
                    'rank': i + 1,
                    'name': pokemon_names[idx],
                    'similarity': float(similarities[idx]),
                    'data': pokemon_docs[idx]
                }
                results.append(result)
        
        print(f"RAG: Found {len(results)} similar Pokemon")
        for result in results:
            print(f"  {result['rank']}. {result['name']} [{result['data']['types'][0]}] (similarity: {result['similarity']:.3f})")
            if "flavorText" in result['data']:
                print(f"   {result['data']['flavorText']}")
            else:
                print(f"   No flavor text available")
            if "attacks" in result['data']:
                attacks = ""
                for attack in result['data']['attacks']:
                    attacks += attack['name'] + " | "
                print(f"  Attacks: {attacks}")
            if "abilities" in result['data']:
                abilities = ""
                for ability in result['data']['abilities']:
                    abilities += ability['name'] + " | "
                print(f"  Abilities: {abilities}")
        
        return results
        
    except Exception as e:
        print(f"Error in RAG function: {str(e)}")
        return []

def procedural_generator():
    pokeDescription = {
        "pokeName": "",
        "pokeType": "",
        "pokeDexEntry": "",
        "pokeDescription_visual": "",
        "pokeDescription_pose": "",
        "pokeDescription_environment": "",
    }
    return pokeDescription

if __name__ == '__main__':
    # TODO: Change these parameters to generate different Pokémon cards
    # TODO: For mechanical generation
    pokeName = "Ori"
    pokeType = "Darkness" # One string of the typeIDs: Grass, Fire, Water, Lightning, Psychic, Fighting, Dark, Darkness, Metal, Fairy, Dragon, Colorless
    pokeDexEntry = "A small feline predator that scratches everything with his claws, but can purr to calm friends."
    model = "qwen/qwen3-14b"
    systemPrompt = 'You are a Pokémon Card Generator. Complete the JSON that I started with the fields "hp", "abilities", "attacks", "resistances", "weaknesses", and "retreatCost". The field "ability" must have a field "text". Strictly follow the specified JSON format. Respond only with valid JSON. Do not write an introduction or summary. Field values cannot be empty or "".'
    use_RAG = True
    num_RAG_samples = 5
    temperature = 2

    # TODO: For visual generation
    pokeDescription_visual = "Dark gray cat with black nose, white paws and white tail tip, fierce, smirk, jumping"
    pokeDescription_pose = "jumping, battle pose"
    pokeDescription_environment = "forest background"

    # TODO: For final output
    author = "Johannes Pfau" # replace with your name

    print("Generating Pokémon card for:", pokeName)

    # Step 1: LLM Prompt
    pokeTCGDict = {
        "id": "aicg2025-1",
        "name": pokeName,
        "flavorText": pokeDexEntry,
        "types": [pokeType],
        "supertype": "Pok\u00e9mon",
        "subtypes": ["Basic"]
    }
    LMStudio_server_address = 'http://localhost:8888/'
    listAvailableModels(LMStudio_server_address)
    promptLMStudio(pokeName, pokeTCGDict, model, systemPrompt, LMStudio_server_address, use_RAG=use_RAG, num_RAG_samples=num_RAG_samples, temperature=temperature)

    # Step 2: Diffusion Prompt
    diffusion_prompt = f"Pokémon, pokemon, {pokeName}, {pokeType} Type, creature, {pokeDexEntry}, \n{pokeDescription_visual}, \n{pokeDescription_pose}, \n{pokeDescription_environment}"
    steps = 20
    cfg = 3.5
    denoise = 1
    lora_strength_niji = 1
    lora_strength_pokemon = 0.4
    workflow_template = 'pokemon-niji-F1.json'
    comfyUI_server_address = 'http://95.99.38.192:8000/'
    # promptComfyUI(diffusion_prompt, workflow_template, steps, cfg, denoise, lora_strength_niji, lora_strength_pokemon, comfyUI_server_address, randomize_seed=True)
