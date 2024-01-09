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

    model = CLIPVisionModelWithProjection.from_pretrained(model_path).to(device)
    processor = CLIPImageProcessor.from_pretrained(processor_path)
    # model = CLIPVisionModelWithProjection.from_pretrained("/home/vtd/scratch/StableDiffusion/CLIPVision")
    # processor = CLIPImageProcessor.from_pretrained("/home/vtd/scratch/StableDiffusion/Processor")

    return model, processor


def get_embed(path, model, processor):

    image = Image.open(path)
    inputs = processor(text=None, images=image, return_tensors="pt")
    # print(type(inputs['pixel_values']))

    pixel_values = inputs['pixel_values'].to('cuda')

    outputs = model(pixel_values)

    image_features = outputs.image_embeds
    image_embeddings = torch.Tensor.cpu(image_features).detach().numpy()[0, :]
    print(image_features.size())

    return image_embeddings


def batch(mat, image_paths, model_path, processor_path):
    model, processor = prep_models(model_path, processor_path)

    for i in range(len(image_paths)):
        image_embeddings = get_embed(image_paths[i], model, processor)
        mat[i, :] = image_embeddings

    return mat


def avg_embeds(embeds):
    avg_embed = np.mean(embeds, 0)
    print(np.size(avg_embed))

    return avg_embed


def get_extreme_features(embeds, sd_num=3):
    mean = np.mean(embeds)
    sd = np.std(embeds)
    indices = np.where((embeds >= (mean + (sd_num * sd)))
                       | (embeds <= mean - (sd_num * sd)))
    return indices[0]


if __name__ == "__main__":

    # path = "test_images/snake.PNG"

    # glob_str = "../animal-images/*.png"
    glob_str = "../imagenet_animals/*.JPEG"


    image_paths = glob.glob(glob_str)
    x = np.zeros((len(image_paths), 768))

    model_path = "../stable-diffusion-2-1-unclip-small/image_encoder"
    processor_path = "../stable-diffusion-2-1-unclip-small/feature_extractor"
    model, processor = prep_models(
        model_path=model_path, processor_path=processor_path)

    # for i in range(len(image_paths)):
    #     image_embeddings = get_embed(image_paths[i], model, processor)
    #     x[i, :] = image_embeddings
    #     print(f"{i}/{len(image_paths)}")

    # np.savetxt('all_embeddings.txt', x, delimiter=',')

    # x = get_embed(path, model, processor)
    # np.savetxt('embed.txt', x, delimiter=',')

    # # Create a histogram
    # plt.hist(image_embeddings, bins=20, color='blue', edgecolor='black')

    # # Add labels and a title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')

    # # Show the plot
    # plt.show()

    for i in range(len(image_paths)):
        image_embeddings = get_embed(image_paths[i], model, processor)
        x[i, :] = image_embeddings
        print(f"{i}/{len(image_paths)}")

    np.savetxt('imagenet_embeddings.txt', x, delimiter=',')
