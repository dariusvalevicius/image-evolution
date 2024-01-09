import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  # Import the Image module from the Pillow library
from glob import glob

# Function to load and resize images
def load_and_resize_image(file_path, target_size=(128, 128)):
    image = Image.open(file_path)
    image = image.resize(target_size)
    return np.array(image)

# Replace these file paths with the actual paths to your images
image_set_paths = [
    "saved_images/Test2/iter*.png",
    "saved_images/Test3/iter*.png",
    "saved_images/Test4/iter*.png",
]

# Load real images from file paths
image_sets = [np.array([load_and_resize_image(image_path) for image_path in glob(set_paths)]) for set_paths in image_set_paths]

# Create a 3x3 grid of subplots
fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(12, 5))

# Flatten the 2D array of subplots for easier indexing
axes = axes.flatten()

set_labels = ["Shark", "Beaver", "Chimpanzee"]

# Loop through each subplot and set the corresponding image from each set
for i, ax in enumerate(axes):
    set_index = i // 10  # Determine which set to use based on subplot index
    image_index = i % 10  # Determine the index within the set
    ax.imshow(image_sets[set_index][image_index])
    ax.axis("off")  # Turn off axis labels

    # Add title only for the first subplot in each row
    if i % 10 == 0:
        ax.set_title(set_labels[set_index])


fig.supxlabel('Top performing image from each generation')
fig.supylabel('Animal target')
fig.suptitle('Testing evolution with ViT classifier')
# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
