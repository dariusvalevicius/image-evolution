
# import requests
import torch
# from PIL import Image
# from io import BytesIO
import numpy as np

from diffusers import StableUnCLIPImg2ImgPipeline
from diffusers.utils import load_image

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('Using device:', device)


pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
    "../stable-diffusion-2-1-unclip-small") #, torch_dtype=torch.float16, variation="fp16")
# pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
#     "/home/vtd/scratch/StableDiffusion/unCLIP_model", torch_dtype=torch.float16, variation="fp16"
# )

pipe = pipe.to(device)
# pipe.enable_model_cpu_offload()
# pipe.enable_vae_slicing()


# embedding = torch.randn(1, 768, dtype=torch.float16)
embedding = torch.tensor(np.reshape(np.loadtxt("avg_embeds.txt", dtype="float16"), (1,768)))
print(embedding.size())
embedding = embedding.to(device)

images = pipe(image_embeds=embedding, num_inference_steps=15).images
images[0].save("snake_avg_embedding.png")
