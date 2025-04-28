import torch
import numpy as np

from diffusers import StableUnCLIPImg2ImgPipeline
import os
import time
from PIL import Image

import pickle
import json
import argparse

import fitness_function


def generate_image(pipe, embedding, image_name, diffusion_steps=21):
    '''
    Generate an image in Stable UnCLIP using a latent embedding.

    Inputs:
        pipe: StableUnCLIPImg2ImgPipeline object
        embedding: n-dimensional numpy array
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

    Input
        model_path: Path to Stable unCLIP model
    Output
        pipe: Stable unCLIP pipeline object
    '''
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16).to('cuda')

    return pipe

def get_embed(path, vision_model, processor):
    '''
    Gets the CLIP embeddings from an image file.

    Inputs
        path: Path to image file
        vision_model: Pipe object for encoding image
        processor: Object for preprocessing image
    Output
        image_embeddings: n-dimensional image encoding
    '''
    
    image = Image.open(path)
    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs['pixel_values'].to('cuda')

    outputs = vision_model(pixel_values)

    image_features = outputs.image_embeds
    image_embeddings = torch.Tensor.cpu(image_features).detach().numpy()[0, :]

    return image_embeddings


def create_new_latents(fitness, latents, mutation_size):
    '''
    This function takes the fitness scores and embeddings of the previous
    generation in order to produce a new set of embeddings.

    Algorithm:
        The fitness values are sorted and the top half is selected.
        The median (of the original fitness vector) is subtracted.
        Finally, the values are normalized (sum to one) to produce recombination weights for the top 1/2 images.
        New latents are samples around the weighted mean.

    Inputs
        fitness: Fitness values computed by fitness function
        latents: The latent embeddings of the images
        mutation_size: The sigma value to use when resampling new latents
    Output
        next_latents: Latent vectors for the next generation
    '''

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
    This script is meant to be run as an asynchronous subprocess companion to the psychopy script.
    If run on one computer, set relevant param in config.json and the psychopy script will automatically start it.
    If run on two computers, start manager.py on analysis computer and the psychopy program on the stimulus presentation computer.

    This script does several things (in this order):
    - Computes initial random latents
    - Generates images from latents
    - Re-encodes images (set "use_post_encodings" to 1 to use these during the real-time evolution)
    - Waits for the onset times to be written from the psychopy program
    - Calls fitness function to compute image fitness
    - Recombines latent embeddings for next generation

    Inputs
        root_dir: The path to this run's data folder
        condition: The experiment condition, dictates which fitness function to use
    '''

    print("GENERATOR is starting...")

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="Root directory of output (run level)")
    parser.add_argument("--condition", help="Experiment current condition/target")
    args = parser.parse_args()

    root_dir = args.output
    condition = args.condition


    with open('../config.json') as f:
        config = json.load(f)

    diffusion_steps = config["diffusion_steps"]
    use_post_encodings = config=["use_post_encodings"]

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

        print(f"GENERATOR: Processing generaton {iter}.")


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
        fitness = fitness_function.get_fitness_scores(root_dir, onset_times, start_time, ratings, condition)

        # Save fitness values
        fitness_path = os.path.join(gen_folder, "fitness.txt")
        np.savetxt(fitness_path, fitness, delimiter=',')


        # Concatenate new generation embeddings
        if use_post_encodings:
            all_latents[iter, :, :] = pca.transform(embeddings_post)
        else:
            all_latents[iter, :, :] = this_trial_latents

        print(f"GENERATOR: Generation {iter} processing complete. Moving on...")

print("GENERATOR finished processing. Closing...")


