import torch
import numpy as np
from diffusers import StableUnCLIPImg2ImgPipeline


def generate_image(pipe, embedding, image_name):
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print('Using device:', device)

    pipe = pipe.to(device)
    # pipe.enable_model_cpu_offload()
    # pipe.enable_vae_slicing()

    embedding = torch.tensor(np.reshape(
        embedding, (1, np.size(embedding))), dtype=torch.float16)
    # print(embedding.size())
    embedding = embedding.to(device)

    images = pipe(image_embeds=embedding, num_inference_steps=21).images
    images[0].save(image_name)


def prep_model(model_path):
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16)

    return pipe


if __name__ == "__main__":

    model_path = "../stable-diffusion-2-1-unclip-small"
    embedding = np.loadtxt("final.txt")
    image_name = "snake_avg_embedding.png"

    pipe = prep_model(model_path)
    generate_image(pipe, embedding, image_name)
