from argparse import ArgumentParser as AP
import torch
from utils import load_dataset
from diffusers.utils import load_image
from sane_pipeline import SANEPipeline
from tqdm import tqdm
from diffusers import EulerAncestralDiscreteScheduler
import os

def main(opts):
    dataset = load_dataset(opts.ds_path, model='gpt')
    # log test params
    os.makedirs(opts.output_dir, exist_ok=True)
    with open(os.path.join(opts.output_dir, 'settings.txt'), 'w+') as file:
        file.write(f'''
        guidance_scale: {opts.guidance_scale}\n
        image_guidance_scale: {opts.image_guidance_scale}\n
        additional_guidance_scale: {opts.additional_guidance_scale}\n
        num_inference_steps: {opts.num_inference_steps}
        ''')
    pipe = SANEPipeline.from_pretrained(opts.model_id,safety_checker=None).to('cuda')

    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    for index, elem in tqdm(enumerate(dataset)):
        if opts.max_images > 0 and index == opts.max_images:
            print(f"Max images {opts.max_images} reached")
            exit()
        os.makedirs(os.path.join(opts.output_dir, '{:03}'.format(index)), exist_ok=True)

        image_path = elem['image_path']
        prompt_edit = elem['instruction']
        additional_instructions = elem['decomposed']
        additional_instructions = additional_instructions[:opts.num_additional_instructions]

        image = load_image(image_path).resize((opts.resolution, opts.resolution))

        for i in range(0, opts.num_samples):
            edited_image = pipe(
                prompt=prompt_edit,
                image=image,
                height=opts.resolution,
                width=opts.resolution,
                additional_instructions=additional_instructions,
                guidance_scale=opts.guidance_scale,
                additional_guidance_scale=opts.additional_guidance_scale,
                image_guidance_scale=opts.image_guidance_scale,
                num_inference_steps=opts.num_inference_steps,
            ).images[0]
            edited_image.save(os.path.join(opts.output_dir, '{:03}'.format(index), '{:06}.png'.format(i)))


if __name__ == '__main__':
    ap = AP()
    ap.add_argument('--ds_path', default='../emuedit_dataset/')
    ap.add_argument('--resolution', default=512, type=int)
    ap.add_argument('--output_dir', default='results/sane/instructpix2pix')
    ap.add_argument('--model_id', default='timbrooks/instruct-pix2pix')
    ap.add_argument('--num_additional_instructions', default=3, type=int)
    ap.add_argument('--num_samples', default=1, type=int)
    ap.add_argument('--guidance_scale', default=7.5, type=float)
    ap.add_argument('--additional_guidance_scale', default=7.5, type=float)
    ap.add_argument('--image_guidance_scale', default=1.5, type=float)
    ap.add_argument('--num_inference_steps', default=30, type=int)
    ap.add_argument('--max_images', default=-1, type=int)
    main(ap.parse_args())
