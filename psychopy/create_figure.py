import matplotlib.pyplot as plt
import numpy as np
import os



def plot_score(file):
    data = np.loadtxt(file, delimiter=",")

    m, n = data.shape

    # Calculate means and standard deviations (for error bars)
    means = np.mean(data, axis=1)
    std_devs = np.std(data, axis=1)

    # Plot the raw data points
    plt.plot(data, 'o', alpha=0.5, color="black")

    # Plot the means with error bars
    plt.errorbar(np.arange(m), means, yerr=std_devs, fmt='o-', color='black')

    plt.xticks(np.arange(m), (np.arange(m)+1).astype(int))
    plt.yticks(np.arange(11))

    # Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.title('Scores by generation')

    # plt.show()

def plot_both(file1, file2):
    data1 = np.loadtxt(file1, delimiter=",")
    m, n = data1.shape

    # Calculate means and standard deviations (for error bars)
    means = np.mean(data1, axis=1)
    std_devs = np.std(data1, axis=1)

    fig, ax1 = plt.subplots()

    # Plot the raw data points
    ax1.plot(data1, 'o', alpha=0.5, color="blue")

    # Plot the means with error bars
    ax1.errorbar(np.arange(m), means, yerr=std_devs, fmt='o-', color='blue')

    # Set x-axis ticks to integers
    ax1.set_xticks(np.arange(m), (np.arange(m)+1).astype(int))

    # Set the range of the y-axis for the left y-axis
    ax1.set_ylim([0, 10])  # Change the range as needed

    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Rating (10-point scale)', color='blue')
    ax1.tick_params('y', colors='blue')

    data2 = np.loadtxt(file2, delimiter=",")
    # Calculate means and standard deviations (for error bars)
    means = np.mean(data2, axis=1)
    std_devs = np.std(data2, axis=1)

    ax2 = ax1.twinx()
    # Plot the raw data points
    ax2.plot(data2, 'o', alpha=0.5, color="orange")

    # Plot the means with error bars
    ax2.errorbar(np.arange(m), means, yerr=std_devs, fmt='o-', color='orange')

    ax2.set_ylabel('SCR (miro-siemens)', color='orange')
    ax2.tick_params('y', colors='orange')

def plot_top_images(file):
    data = np.loadtxt(file, delimiter=",")

    top_image_indices = np.argmax(data, axis=1)

    directory = os.path.dirname(file)
    # Assuming you have a list of image paths (replace this with your own paths)
    # Construct file names based on indices
    image_paths = []
    for i in range(len(top_image_indices)):
        image_paths.append(f"{directory}/images/generation_{i}/num-{top_image_indices[i]}.PNG")

    # Create a figure and axes for each image
    fig, axes = plt.subplots(1, len(image_paths), figsize=(8, 2))

    # Iterate through image paths and plot each image with a number
    for i, image_path in enumerate(image_paths):
        # Load and plot the image
        img = plt.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide axes for better visualization

    fig.suptitle("Top images from each generation")
    # Adjust layout to prevent overlap
    plt.tight_layout()


if __name__ == "__main__":

    plot_top_images("data/780842/selfreport/ratings.txt")
    plt.show()

    plot_both("data/780842/selfreport/ratings.txt", "data/780842/selfreport/scr_scores.txt")
    plt.title('Image evolution: Self-Report Block')
    plt.show()
    plot_both("data/747550/scr/ratings.txt", "data/747550/scr/scr_scores.txt")
    plt.title('Image evolution: SCR Block')
    plt.show()

