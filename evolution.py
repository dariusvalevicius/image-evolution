print("------------ Starting Evolution Pipeline ------------")
print("Importing packages...")
import sys
sys.path.append('/home/vtd/scratch/StableDiffusion/Darius')

import get_embeddings, generate_code, unCLIP_evolution
from glob import glob
import numpy as np

print("Running...")

# vec_size = 768
vec_size = 1024
max_iters = 500

# model_path = "../stable-diffusion-2-1-unclip-small/image_encoder"
# processor_path = "../stable-diffusion-2-1-unclip-small/feature_extractor"
model_path = "/home/vtd/scratch/StableDiffusion/CLIPVision"
processor_path = "/home/vtd/scratch/StableDiffusion/Processor"

# model_path = "../stable-diffusion-2-1-unclip-small"
unclip_model_path = "/home/vtd/scratch/StableDiffusion/unCLIP_model"

##### Protocol 1: Generate images at different iteration steps

##############################################################
##### Avg of snake embeddings
# # Get reference image embeddings
# glob_str = "test_images/snakes/*.png"

# image_paths = glob(glob_str)
# x = np.zeros((len(image_paths), vec_size))

# print("Getting embeddings...")
# x = get_embeddings.batch(mat=x, image_paths=image_paths, model_path=model_path, processor_path=processor_path)

# # np.savetxt('embeds_all.txt', x, delimiter=',')

# # Average embeddings
# x = get_embeddings.avg_embeds(x)
# # x = np.reshape(np.loadtxt("embed.txt", dtype="float16"), (1,768))

###################################################################
#### Single snake embedding
# image_path = ["test_images/bat.png"]
# x = np.zeros((1, vec_size))

# x = get_embeddings.batch(mat=x, image_paths=image_path, model_path=model_path, processor_path=processor_path)

# #### Run evolution
# print("Running evolution...")
# err_out, x_out = generate_code.run_evolution(
#     target=x, 
#     mutation_rate=0.02, 
#     mutation_size=0.75, 
#     max_iters=max_iters, 
#     error_power=6,
#     vec_size=vec_size)


# # Generate image for every n iterations

# print("Generating images...")
# for i in range(max_iters):
#     if (i == 0) or ((i+1) % 50 == 0):

#         embedding = x_out[i,:]
#         image_name = f"output_images/iteration_{i+1}.png"
#         unCLIP_evolution.generate_image(model_path=unclip_model_path, embedding=embedding, image_name=image_name)
###########################################

############################################
###### Generate images with features removed

image_path = ["test_images/snake.png"]
x = np.zeros((1, vec_size))

x = get_embeddings.batch(mat=x, image_paths=image_path, model_path=model_path, processor_path=processor_path)
embedding = x[0,:]

indices = get_embeddings.get_extreme_features(embedding, sd_num=3)
print(f"Extreme features: {indices}")


print("Generating images...")
image_name = "output_images/base.png"
unCLIP_evolution.generate_image(model_path=unclip_model_path, embedding=embedding, image_name=image_name)

for index in indices:
    new_embed = embedding
    new_embed[index] = 0

    image_name = f"output_images/feature_{index}_knockout.png"
    unCLIP_evolution.generate_image(model_path=unclip_model_path, embedding=new_embed, image_name=image_name)

image_name = "output_images/all_knockout.png"
new_embed = embedding
new_embed[indices] = 0
unCLIP_evolution.generate_image(model_path=unclip_model_path, embedding=embedding, image_name=image_name)







