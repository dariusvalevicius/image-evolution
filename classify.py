from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import glob
import numpy as np
import torch
import shutil
import os

# Define parameters
imagenet_indices = {
    "rabbit": 330,
    "cockroach": 314,
    "gecko": 38,
    "spider": 76,
    "chicken": 137,
    "grasshopper": 311,
    "butterfly": 323,
    "bird": 15,
    "peacock": 84,
    "dog": 207,
    "cat": 281,
    "snake": 61,
    "fish": 391,
    "frog": 31,
    "turtle": 37,
    "beetle": 305,
    "ant": 310,
    "bee": 309,
    "guinea pig": 338,
    "sheep": 348,
    "shark": 2,
    "whale": 147
}


def return_score(processor, model, image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    pixel_values = inputs['pixel_values'].to('cuda')
    outputs = model(pixel_values)
    logits = torch.Tensor.cpu(outputs.logits).detach().numpy()

    # model predicts one of the 1000 ImageNet classes
    # for i in range(5):
    # predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", model.config.id2label[predicted_class_idx])
    # print("Predicted class:", model.config.id2label[i])

    # i = 2  # Let's go with great white shark
    # print(f"Predicted class: {model.config.id2label[i]}")
    # print(f"Logit score: {logits[0,i]}")

    return logits


def prep_model(path):

    processor = ViTImageProcessor.from_pretrained(path)
    model = ViTForImageClassification.from_pretrained(path).to('cuda')

    return processor, model


if __name__ == "__main__":

    processor, model = prep_model('../vit-base-patch16-224')

    # Get a list of all entries (files and directories) in the folder
    entries = os.listdir('sim')

    # Filter out only the directories
    directories = [entry for entry in entries if os.path.isdir(os.path.join('sim', entry))]



    for directory in directories:
        x = np.zeros((10,8)) 

        for i in range(10):

            image_paths = glob.glob(f"sim/{directory}/generation_{i}/*.PNG")
            for j in range(len(image_paths)):
                logits = return_score(processor, model, image_paths[j])[:,:398]
                probabilities = np.exp(logits) / np.sum(np.exp(logits))
                # print(probabilities.shape)
                x[i, j] = probabilities[0, imagenet_indices[directory]]

        print(f"Finished: {directory}")        

        np.savetxt(f'sim/{directory}/probabilities.txt', x, delimiter=',')




    # glob_str = "../imagenet_animals/*.JPEG"

    # image_paths = glob.glob(glob_str) # First 10,000 images
    # x = np.zeros((len(image_paths), 1000))    

    # for i in range(len(image_paths)):
    #     score = return_score(processor, model, image_paths[i])
    #     x[i, :] = score
    #     print(f"{i}/{len(image_paths)}")

    # np.savetxt('all_logits.txt', x, delimiter=',')

    # j = 0

    # for i in range(len(image_paths)):
    #     score = return_score(processor, model, image_paths[i])
    #     x[i, :] = score
    #     if i % 100 == 0:
    #         print(f"{i}/{len(image_paths)}")

    #     if np.argmax(score) < 398:
    #         shutil.copyfile(image_paths[i], f"../imagenet_animals/imagenet-animal-{j}.JPEG")
    #         # print(f"{np.argmax(score)}")
    #         j = j + 1


    # # Find the maximum value in each row
    # max_indices = np.argmax(x, axis=1)
    
    # # Find the indices of rows where the maximum value is within the first n columns
    # indices = np.where(max_indices < 398)[0]
    
    # # Subset the array based on the indices
    # result = x[indices, :]

    # np.savetxt('imagenet_logits.txt', x[:,:398], delimiter=',')

