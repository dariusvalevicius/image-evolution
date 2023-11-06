# import requests
import torch
# from PIL import Image
# from io import BytesIO
import numpy as np

from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

def generate_image(model_path, embedding, image_name):
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Using device:', device)

    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(model_path, torch_dtype=torch.float16, variation="fp16")
    # pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    #     "/home/vtd/scratch/StableDiffusion/unCLIP_model", torch_dtype=torch.float16, variation="fp16"
    # )

    pipe = pipe.to(device)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()

    # embedding = torch.randn(1, 768, dtype=torch.float16)
    embedding = torch.tensor(np.reshape(embedding, (1,np.size(embedding))), dtype=torch.float16)
    print(embedding.size())
    embedding = embedding.to(device)

    images = pipe(image_embeds=embedding, num_inference_steps=21).images
    images[0].save(image_name)

if __name__ == "__main__":

    model_path = "../stable-diffusion-2-1-unclip-small"
    embedding = np.loadtxt("final.txt")
    image_name = "snake_avg_embedding.png"

    generate_image(model_path, embedding, image_name)