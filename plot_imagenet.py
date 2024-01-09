import matplotlib.pyplot as plt
import numpy as np


def plot_max_indices_histogram(matrix):
    # Find the indices of the max values for each row
    max_indices = np.argmax(matrix, axis=1)
    
    # Create a histogram of the max indices
    plt.hist(max_indices, bins=np.arange(matrix.shape[1] + 1) - 0.5, edgecolor='black')
    
    # Customize the plot
    plt.title('Histogram of Max Indices in Each Row')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Show the plot
    plt.show()

logits = np.loadtxt("imagenet_logits.txt", delimiter=',')


plot_max_indices_histogram(logits[:,:398])