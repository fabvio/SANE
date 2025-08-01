from utils import load_dataset
import os
from argparse import ArgumentParser as AP
from prompts import prompt_caption_gpt
from utils import call_gpt
import io
from PIL import Image
import base64
from tqdm import tqdm

def caption(dataset_path):
    # Step one: read the input image and get the caption

    dataset = load_dataset(dataset_path)

    # log test params
    os.makedirs(os.path.join(dataset_path, 'captions_gpt'), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, 'captions_updated'), exist_ok=True)

    for index, elem in tqdm(enumerate(dataset)):
        image_path = elem['image_path']
        instruction = elem['instruction']
        prompt = prompt_caption_gpt.format(instruction)
        image = Image.open(image_path).resize((256, 256)).convert('RGB')
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()

        # Encode the image as a base64 string
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        caption = call_gpt(prompt, image=image_base64, model='gpt-4-turbo')
        caption_original = caption.split('\n')[0][2:].strip()
        caption_updated = caption.split('\n')[1][2:].strip()

        with open(os.path.join(dataset_path, 'captions_gpt', '{:06}.txt'.format(index)), 'w+') as file:
            file.write(caption_original)
        with open(os.path.join(dataset_path, 'captions_updated', '{:06}.txt'.format(index)), 'w+') as file:
            file.write(caption_updated)

if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--ds_path', required=True)
    args = ap.parse_args()
    caption(args.ds_path)
