from diffusers import StableDiffusionXLPipeline
import torch

pipe = StableDiffusionXLPipeline.from_pretrained(
    'stabilityai/stable-diffusion-xl-base-1.0',
    torch_dtype=torch.float16,
    variant='fp16'
).to('cuda')

def generate(prompt):
    image = pipe(prompt=prompt).images[0]
    path = '/tmp/output.png'
    image.save(path)
    return path

def handler(event):
    prompt = event['input']['prompt']
    img_path = generate(prompt)

    return {
        'ok': True,
        'mode': 'sdxl_real',
        'output_image_path': img_path
    }
