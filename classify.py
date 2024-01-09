from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import glob
import numpy as np
import torch
import shutil


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

    glob_str = "../imagenet_animals/*.JPEG"

    image_paths = glob.glob(glob_str) # First 10,000 images
    x = np.zeros((len(image_paths), 1000))    

    for i in range(len(image_paths)):
        score = return_score(processor, model, image_paths[i])
        x[i, :] = score
        print(f"{i}/{len(image_paths)}")

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

    np.savetxt('imagenet_logits.txt', x[:,:398], delimiter=',')

