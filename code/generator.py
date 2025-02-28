import torch
import numpy as np
import pandas as pd

from diffusers import StableUnCLIPImg2ImgPipeline
import sys
import os
import time
from PIL import Image

import re

import pickle
import json

from datetime import datetime, timedelta

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import fitness_function



def generate_image(pipe, embedding, image_name, diffusion_steps=21):
    '''
    Generate an image in Stable UnCLIP using a latent embedding.
        Inputs:
            pipe: StableUnCLIPImg2ImgPipeline object
            embedding: 1024-d numpy array
            image_name: output filename
            diffusion_steps: Number of diffusion steps for image generation. Can tweak this based on compute power.
        Outputs:
            Saved image.
    '''

    embedding = torch.tensor(np.reshape(
        embedding, (1, np.size(embedding))), dtype=torch.float16)
    embedding = embedding.to('cuda')

    images = pipe(image_embeds=embedding, num_inference_steps=diffusion_steps).images
    images[0].save(image_name)
    return None


def prep_model(model_path):
    '''
    Helper function to load Stable UnCLIP model into memory.
    '''
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16).to('cuda')

    return pipe

def get_embed(path, vision_model, processor):
    '''
    Gets the CLIP embeddings from an image file.
    '''
    
    image = Image.open(path)
    inputs = processor(text=None, images=image, return_tensors="pt")

    pixel_values = inputs['pixel_values'].to('cuda')

    outputs = vision_model(pixel_values)

    image_features = outputs.image_embeds
    image_embeddings = torch.Tensor.cpu(image_features).detach().numpy()[0, :]

    return image_embeddings


def create_new_latents(fitness, latents, mutation_size):
    '''This function takes the fitness scores and embeddings of the previous
    generation in order to produce a new set of embeddings'''

    top_n = int(fitness.shape[0]/2)
    
    # Get top vectors
    idx = np.argsort(fitness)[::-1]
    fitness_sorted = fitness[idx]
    latents_sorted = latents[idx, :]

    fitness_top = fitness_sorted[:top_n]
    latents_top = latents_sorted[:top_n, :]

    # Compute recombination weights
    median = np.median(fitness)
    fitness_relative = np.clip(fitness_top - median, 0, None)
    
    if np.sum(fitness_relative) == 0:
        weights = np.ones(top_n) / top_n
    else:
        weights = fitness_relative / np.sum(fitness_relative)

    mean = np.sum((latents_top.T * weights).T, axis=0)
    next_latents = np.random.multivariate_normal(mean, mutation_size * np.eye(latents.shape[1]), size=fitness.shape[0])
    
    return next_latents



if __name__ == "__main__":

    '''

    '''

    print("GENERATOR is starting...")

    # Set start time


    # Get arguments
    root_dir = sys.argv[1]

    with open('../config.json') as f:
        config = json.load(f)

    diffusion_steps = config["diffusion_steps"]

    pop_size = config["pop_size"]
    max_iters = config["max_iters"]

    latent_size = config["latent_size"]
    embedding_size = config["embedding_size"]
    mutation_size = config["mutation_size"]

    pca_path = config["pca_path"]
    model_path = config["unclip_model_path"]

    # Load PCA model
    with open(os.path.join("..", pca_path), 'rb') as f:
        pca = pickle.load(f)

    # Prepare generator model
    pipe = prep_model(model_path)
    vision_model = pipe.image_encoder
    processor = pipe.feature_extractor

    embeddings_files_complete = []
    onset_files_complete = []

    all_latents = np.zeros((max_iters, pop_size, latent_size))
    fitness = 0

    ## Start loop
    for iter in range(max_iters):

        # Create folder for this generation
        gen_folder = os.path.join(root_dir, f"generation_{iter:02}")
        if not os.path.exists(gen_folder):
            os.makedirs(gen_folder)
        
        # Generate vectors
        if iter == 0:
            this_trial_latents = np.random.multivariate_normal(np.zeros(latent_size), np.eye(latent_size), size=pop_size)    
        else:
            latents = all_latents[iter - 1, :, :]
            this_trial_latents = create_new_latents(fitness, latents, mutation_size)

        # Concatenate new generation embeddings
        all_latents[iter, :, :] = this_trial_latents

        # Generate batch of image embeddings
        embeddings = pca.inverse_transform(this_trial_latents)

        # Save latents
        latents_path = os.path.join(gen_folder, "latents.txt")
        np.savetxt(latents_path, this_trial_latents, delimiter=',')

        embeddings_post = np.zeros_like(embeddings)
        finished = np.zeros(pop_size)

        for i in range(pop_size):

            filename = os.path.join(gen_folder, f"img_{i:02}.png")
            # Generate image
            generate_image(pipe, embeddings[i,:], filename, diffusion_steps=diffusion_steps) ## Edit diffusion steps based on performance
            # Read back embeddings
            this_embedding_post = get_embed(filename, vision_model, processor)
            embeddings_post[i,:] = this_embedding_post

            # Set status to complete
            finished[i] = 1
            np.savetxt(os.path.join(gen_folder, "status.txt"), finished, delimiter=',')

        # Save post embeddings
        np.savetxt(os.path.join(gen_folder, "embeddings_post.txt"), embeddings_post, delimiter=',')
            
        # Wait for onsets file
        onsets_path = os.path.join(gen_folder, "onset_times.txt")
        while not os.path.isfile(onsets_path):
            time.sleep(1)

        # Check if file is written before loading
        while True:
            try:
                with open(onsets_path, 'rb') as _:
                    _.close()
                    break
            except IOError:
                time.sleep(1)
        
        onset_times = np.loadtxt(onsets_path, delimiter=',')

        # Get ratings as well
        ratings_path = os.path.join(gen_folder, "ratings.txt")
        ratings = np.loadtxt(ratings_path, delimiter = ",")

        # Read in start time
        with open(os.path.join(root_dir, "run_start_time.txt"), 'r') as f:
            start_time = f.read()

        ## Compute new fitness values
        fitness = fitness_function.get_fitness_scores(onset_times, start_time, ratings)


        print(f"GENERATOR: Generation {iter} processing complete. Moving on...")

print("GENERATOR finished processing. Closing...")


