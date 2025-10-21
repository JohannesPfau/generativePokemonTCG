import json
import random
import urllib
import uuid

import websocket # pip install https://github.com/websocket-client/websocket-client/archive/refs/tags/v1.8.0.zip
from pokemonGenerate import pokeName, pokeDescription, pokeType

prompt = "pokemon, " + pokeName + ", " + pokeType + ", creature, " + pokeDescription + ", "
prompt += "((action, battle, combat)), jump, "
prompt += "dark gray cat, (((fierce))), (((smirk))), forest background, " # TODO: Add to prompt any visual characteristics that might benefit the image generation / background
# prompt += "flame body, fire hands, fire legs, burning, ethereal, elemental, atronach, hellfire crucible background, forge, pandemonium"
prompt += "8k unity render, action shot, very dark lighting, heavy shadows, detailed, detailed face, (vibrant, photo realistic, realistic, dramatic, dark, sharp focus, 8k), (intricate:1.4), decadent, (highly detailed:1.4), digital painting, (global illumination, studio light, volumetric light),concept art, smooth, sharp focus, illustration, art by artgerm,(loish:0.23),octane render, artstation"
negativePrompt = "white background, (((human))), (((skin))), (((nsfw)))"
negativePrompt += "extra limbs, extra legs, disfigured, deformed, mutated hands and fingers, out of frame, long neck, (bad art, low detail, pencil drawing:1.4), (plain background, grainy, low quality), (watermark, thin lines:1.2), (deformed, signature:1.2), (blurry, ugly, bad anatomy, extra limbs, undersaturated, low resolution), deformations, out of frame, amputee, bad proportions, extra limb, missing limbs, distortion, floating limbs, out of frame, text, malformed, cropped"

def open_websocket_connection():
  server_address='127.0.0.1:8188'
  client_id=str(uuid.uuid4())
  ws = websocket.WebSocket()
  ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
  return ws, server_address, client_id

def queue_prompt(prompt, client_id, server_address):
  p = {"prompt": prompt, "client_id": client_id}
  headers = {'Content-Type': 'application/json'}
  data = json.dumps(p).encode('utf-8')
  req =  urllib.request.Request("http://{}/prompt".format(server_address), data=data, headers=headers)
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
                      print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')
          if message['type'] == 'executing':
              data = message['data']
              if data['node'] not in finished_nodes:
                  finished_nodes.append(data['node'])
                  print('Progess: ', len(finished_nodes), '/', len(node_ids), ' Tasks done')

              if data['node'] is None and data['prompt_id'] == prompt_id:
                  break #Execution is done
      else:
          continue
  return

def generate_image_by_prompt(p):
  try:
    print(prompt)
    with open(pokeName + '_5_ComfyUIworkflow.json', 'w+') as f:
        json.dump(p, f)
    ws, server_address, client_id = open_websocket_connection()
    prompt_id = queue_prompt(p, client_id, server_address)['prompt_id']
    track_progress(p, ws, prompt_id)
  finally:
    ws.close()
  return

# This is the default workflow of ComfyUI. Modify it directly or via changing dictionary values (see end of script)
workflow_text = """
{
    "3": {
        "class_type": "KSampler",
        "inputs": {
            "cfg": 8,
            "denoise": 1,
            "latent_image": [
                "5",
                0
            ],
            "model": [
                "4",
                0
            ],
            "negative": [
                "7",
                0
            ],
            "positive": [
                "6",
                0
            ],
            "sampler_name": "euler",
            "scheduler": "normal",
            "seed": 8566257,
            "steps": 20
        }
    },
    "4": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {
            "ckpt_name": "v1-5-pruned-emaonly.safetensors"
        }
    },
    "5": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "batch_size": 1,
            "height": 512,
            "width": 512
        }
    },
    "6": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "masterpiece best quality girl"
        }
    },
    "7": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "clip": [
                "4",
                1
            ],
            "text": "bad hands"
        }
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": [
                "3",
                0
            ],
            "vae": [
                "4",
                2
            ]
        }
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "filename_prefix": "ComfyUI",
            "images": [
                "8",
                0
            ]
        }
    }
}
"""

workflow = json.loads(workflow_text)
workflow["3"]["inputs"]["steps"] = "40"
workflow["3"]["inputs"]["seed"] = str(random.randint(0,123456))
workflow["4"]["inputs"]["ckpt_name"] = "sd3_medium_incl_clips.safetensors"
workflow["5"]["inputs"]["height"] = "778"
workflow["5"]["inputs"]["width"] = "1250"
workflow["5"]["inputs"]["batch_size"] = "1"
workflow["6"]["inputs"]["text"] = prompt
workflow["7"]["inputs"]["text"] = negativePrompt

# promptResult = queue_prompt(workflow, client_id, server_address)
generate_image_by_prompt(workflow)
print("DONE! Check your ComfyUI output folder for the image.")