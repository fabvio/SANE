import torch
import os
from utils import call_gpt, load_dataset
from argparse import ArgumentParser as AP
from prompts import prompt_decompose as prompt
from tqdm import tqdm 

def get_instructions(prompt_input, max_attempts, num_instructions):
    while True:
        trials = 0
        try:
            gpt_reply = call_gpt(prompt_input)
            instructions = gpt_reply.split('\n')
            instructions = [x.replace('Suggested output:', '') for x in instructions]
            instructions = [x for x in instructions if x != '']
            print(instructions)
            if len(instructions) != num_instructions:
                trials += 1
                continue
            break
        except KeyboardInterrupt:
            exit()
        except:
            if trials == max_attempts:
                print("Cannot parse: {}".format(gpt_reply))
                exit()
            trials += 1
            continue
    return instructions

def main(args):
    dataset = load_dataset(args.ds_path)
    os.makedirs(os.path.join(args.ds_path, 'decomposed_instructions'), exist_ok=True)

    for index, elem in tqdm(enumerate(dataset)):
        image_path = elem['image_path']
        instruction = elem['instruction']
        caption = elem['caption']
        prompt_input = prompt.format(args.num_instructions, caption, instruction)
        instructions = get_instructions(prompt_input, 5, args.num_instructions)
        instruction_string = ';'.join(instructions)
        filename = os.path.basename(image_path).replace('.png', '.txt').replace('.jpg', '.txt')
        target_path = image_path.replace('original_images', 'decomposed_instructions')
        target_path = os.path.join(os.path.dirname(target_path), filename)
        with open(target_path, 'w+') as file:
            file.write(instruction_string)


if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--ds_path', required=True)
    ap.add_argument('--num_instructions',default=3, type=int)
    args = ap.parse_args()
    main(args)
