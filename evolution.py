import get_embeddings, generate_code, unCLIP
from glob import glob
import numpy as np

if __name__ == "__main__":

    # vec_size = 768
    vec_size = 1024

    # Protocol 1: Generate images at different iteration steps

    # Get reference image embeddings
    glob_str = "test_images/snakes/*.png"

    image_paths = glob(glob_str)
    x = np.zeros((len(image_paths), vec_size))

    # model_path = "../stable-diffusion-2-1-unclip-small/image_encoder"
    # processor_path = "../stable-diffusion-2-1-unclip-small/feature_extractor"
    model_path = "/home/vtd/scratch/StableDiffusion/CLIPVision"
    processor_path = "/home/vtd/scratch/StableDiffusion/Processor"

    print("Getting embeddings...")
    x = get_embeddings.batch(mat=x, image_paths=image_paths, model_path=model_path, processor_path=processor_path)

    # np.savetxt('embeds_all.txt', x, delimiter=',')

    # Average embeddings
    x = get_embeddings.avg_embeds(x)
    # x = np.reshape(np.loadtxt("embed.txt", dtype="float16"), (1,768))


    # Run evolution
    print("Running evolution...")
    err_out, x_out = generate_code.run_evolution(
        target=x, 
        mutation_rate=0.02, 
        mutation_size=1, 
        max_iters=300, 
        error_power=4,
        vec_size=vec_size)
    

    # Generate image for every n iterations

    # model_path = "../stable-diffusion-2-1-unclip-small"
    model_path = "/home/vtd/scratch/StableDiffusion/unCLIP_model"

    print("Generating images...")
    for i in range(300):
        if (i == 1) or (i+1 % 50 == 0):

            embedding = x_out[i,:]
            image_name = f"output_images/iteration_{i}.png"
            unCLIP.generate_image(model_path=model_path, embedding=embedding, image_name=image_name)



