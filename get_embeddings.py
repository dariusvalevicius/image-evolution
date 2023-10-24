import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
import matplotlib.pyplot as plt
import numpy as np
import glob


def prep_models(model_path, processor_path):
    # setting device on GPU if available, else CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    model = CLIPVisionModelWithProjection.from_pretrained(model_path)
    processor = CLIPImageProcessor.from_pretrained(processor_path)
    # model = CLIPVisionModelWithProjection.from_pretrained("/home/vtd/scratch/StableDiffusion/CLIPVision")
    # processor = CLIPImageProcessor.from_pretrained("/home/vtd/scratch/StableDiffusion/Processor")

    return model, processor


def get_embed(path, model, processor):

    image = Image.open(path)
    inputs = processor(text=None, images=image, return_tensors="pt")

    outputs = model(**inputs)

    image_features = outputs.image_embeds
    image_embeddings = image_features.detach().numpy()[0,:]
    print(image_features.size())

    return image_embeddings


def batch(mat, image_paths, model_path, processor_path):
    model, processor = prep_models(model_path, processor_path)

    for i in range(len(image_paths)):
        image_embeddings = get_embed(image_paths[i], model, processor)
        mat[i,:] = image_embeddings

    return mat

def avg_embeds(embeds):
    avg_embed = np.mean(embeds, 0)
    print(np.size(avg_embed))

    return avg_embed

if __name__ == "__main__":

    path = "test_images/snake.PNG"

    glob_str = "test_images/snakes/*.png"

    image_paths = glob.glob(glob_str)
    x = np.zeros((len(image_paths), 768))

    model, processor = prep_models()

    for i in range(len(image_paths)):
        image_embeddings = get_embed(image_paths[i], model, processor)
        x[i,:] = image_embeddings

    

    np.savetxt('embeds_all.txt', x, delimiter=',')

    # # Create a histogram
    # plt.hist(image_embeddings, bins=20, color='blue', edgecolor='black')

    # # Add labels and a title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')

    # # Show the plot
    # plt.show()

