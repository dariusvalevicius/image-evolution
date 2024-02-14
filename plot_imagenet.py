import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
import ast
import json
import seaborn as sns
import pandas as pd


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


if __name__ == "__main__":


    # Step 1: Read the text file
    file_path = 'imagenet_labels.txt'  # Replace with your file path
    with open(file_path, 'r') as file:
        file_content = file.read()

    # Step 2: Convert text to a dictionary
    try:
        label_dict = ast.literal_eval(file_content)
    except (ValueError, SyntaxError) as e:
        print(f"Error while parsing the file content: {e}")
        label_dict = {}

    # Iterate through each key-value pair in the dictionary
    for key, value in label_dict.items():
        # Split the value based on the comma and get the part before the comma
        value_before_comma = value.split(',')[0].strip()

        # Update the value in the dictionary
        label_dict[key] = value_before_comma



    
    data = np.loadtxt("imagenet_logits.txt", delimiter=',')[:,:398]


    # # Create a pandas DataFrame with feature names
    # df = pd.DataFrame(data, columns=list(label_dict.keys())[:398])

    # # Calculate the correlation matrix
    # correlation_matrix = df.corr()


    # Calculate the correlation matrix
    correlation_matrix = np.corrcoef(data, rowvar=False)

    sns.clustermap(correlation_matrix, cmap="coolwarm", method="average", figsize=(10, 8))

    # Display the plot
    plt.show()


    # plot_max_indices_histogram(logits.T)

    # # Perform hierarchical clustering
    # linkage_matrix = linkage(data, method='ward')  # You can choose different linkage methods based on your requirements
    # labels = AgglomerativeClustering(n_clusters=15, linkage='ward').fit_predict(data)

    # # Plot the dendrogram
    # dendrogram(linkage_matrix)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Distance')
    # plt.show()

    # # Print the cluster labels
    # print("Cluster Labels:", labels)



    # # Apply PCA to reduce the dimensionality for visualization
    # pca = PCA(n_components=2)
    # data_pca = pca.fit_transform(data)

    # # Plot the clusters on a PCA plot
    # plt.scatter(data_pca[:, 0], data_pca[:, 1], c=labels, cmap='viridis', alpha=0.5)
    # plt.title('Hierarchical Clustering Results on PCA Plot')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()


    # # Initialize a dictionary to store dimensions for each cluster
    # dimensions_in_clusters = {}

    # # Iterate through each dimension and its corresponding cluster label
    # for dim_index, cluster_label in enumerate(labels):
    #     # Add the dimension index to the list for the corresponding cluster label
    #     dimensions_in_clusters.setdefault(str(cluster_label), []).append(label_dict[dim_index])

    # # Now 'dimensions_in_clusters' is a dictionary where each key is a cluster label,
    # # and the corresponding value is a list of dimensions in that cluster
    # print(dimensions_in_clusters)

    # # Write the dictionary to a JSON file
    # json_file_path = "clusters.json"
    # with open(json_file_path, 'w') as json_file:
    #     json.dump(dimensions_in_clusters, json_file)