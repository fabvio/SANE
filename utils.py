import openai
import time
from PIL import Image
import os
import torch


def set_api_key():
    if os.environ.get("OPENAI_API_TYPE", "none") == "azure":
        openai.api_key      = os.environ.get("OPENAI_API_KEY")
        openai.api_type     = os.environ.get("OPENAI_API_TYPE", "azure")
        openai.api_version  = os.environ.get("OPENAI_API_VERSION", "2023-03-15-preview")
        openai.api_base     = os.environ.get("OPENAI_API_BASE", "")
    else:
        openai.api_key      = os.environ.get("OPENAI_API_KEY")


def call_openai_completion(engine, messages, temperature=1.0, seed=42):
    set_api_key()

    while True:
        try:
            response = openai.ChatCompletion.create(
                model=engine,
                seed=seed,
                messages=messages,
                temperature=temperature,
            )
            response["choices"][0]["message"].to_dict()["content"]
            return response
        except (
                RateLimitError,
                KeyError):

            print("RateLimitError, retrying...")
            time.sleep(2)
            continue
        except Exception as e:
            print("Error:", e)
            exit(-1)

def call_gpt(prompt, image=None, seed=42, temperature=1.0, model='gpt-4o'):

    if image is None:
        messages = [{
            "role": 'system',
            'content': 'You are a helpful assistant.'
        },{
            "role": 'user',
            'content': prompt
        }]
    elif isinstance(image, list):
        image_messages = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{im}"}} for im in image]
        content_message = [{"type": "text", "text": f"{prompt}"}]
        user_message = {"role": "user", "content": content_message + image_messages}
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            user_message
        ]
    else:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [
                {"type": "text", "text": f"{prompt}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}
            }
        ]}]

    reply = call_openai_completion(engine=model, messages=messages, seed=seed)
    return reply.choices[0].message.content

def load_dataset(path, model='gpt'):
    image_path = os.path.join(path, 'original_images')
    paths = []
    image_paths = os.listdir(image_path)
    image_paths.sort()
    for x in image_paths:
        with open(os.path.join(path, 'instructions', x.replace('png', 'txt'))) as file:
            instruction = file.read().splitlines()[0]
        if os.path.exists(os.path.join(path, 'captions_gpt', x.replace('png', 'txt'))):
            with open(os.path.join(path, 'captions_gpt', x.replace('png', 'txt'))) as file:
                caption = file.read().splitlines()[0]
        else:
            caption = None
        if model == 'gpt':
            if os.path.exists(os.path.join(path, 'decomposed_instructions', x.replace('png', 'txt'))):
                with open(os.path.join(path, 'decomposed_instructions', x.replace('png', 'txt'))) as file:
                    decomposed = file.read().splitlines()[0]
                decomposed = decomposed.split(';')
            else:
                decomposed = None
        else:
            try:
                if os.path.exists(os.path.join(path, f'decomposed_instructions_{model}')):
                    with open(os.path.join(path, f'decomposed_instructions_{model}', x.replace('png', 'txt'))) as file:
                        decomposed = file.read().splitlines()[0]
                    decomposed = decomposed.split(';')
                else:
                    decomposed = None
            except:
                decomposed = None
        paths.append({'image_path': os.path.join(image_path, x),
                      'instruction': instruction,
                      'caption': caption,
                      'decomposed': decomposed})
    return paths
