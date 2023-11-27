from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image


def return_score(processor, model, image_path):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits

    # model predicts one of the 1000 ImageNet classes
    # for i in range(5):
    # predicted_class_idx = logits.argmax(-1).item()
    # print("Predicted class:", model.config.id2label[predicted_class_idx])
    # print("Predicted class:", model.config.id2label[i])

    i = 2  # Let's go with great white shark
    # print(f"Predicted class: {model.config.id2label[i]}")
    # print(f"Logit score: {logits[0,i]}")

    return logits[0, i]


def prep_model(path):

    processor = ViTImageProcessor.from_pretrained(path)
    model = ViTForImageClassification.from_pretrained(path)

    return processor, model


# if __name__ == "__main__":

#     processor, model = prep_model('../vit-base-patch16-224')

#     image_path = "test_images/snake.png"
#     score = return_score(processor, model, image_path)
