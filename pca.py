import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap
from sklearn.datasets import make_s_curve
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import ast  # For safely evaluating literal expressions
import unCLIP_evolution
import pickle as pk
import umap

def plot_pca(data):
    # Plot CMA on 2D PCA plot

    pca = PCA(n_components=2)

    # Fit the data
    pca.fit(data)

    # Print the variance explained by the first two dimensions
    explained_variance = pca.explained_variance_ratio_
    print("Variance explained by the first dimension:", explained_variance[0])
    print("Variance explained by the second dimension:", explained_variance[1])

    principal_components = pca.fit_transform(data)

    # The transformed data contains the principal components
    # print("Original data shape:", data.shape)
    # print("Transformed data shape:", principal_components.shape)
    # print("Principal Components:\n", principal_components)

    # Plot the scores on a scatter plot with different colors for each observation
    plt.scatter(
        principal_components[:, 0], principal_components[:, 1], c=np.arange(len(data)), cmap='viridis')

    # Add labels and a colorbar
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Observation Index')

    return None

if __name__ == "__main__":

    embeddings = np.loadtxt("imagenet_embeddings.txt", delimiter=',')
    logits = np.loadtxt("imagenet_logits.txt", delimiter=',')#[:, :398]

    # Find the indices of the max values for each row
    # max_indices = np.argmax(logits, axis=1)
    
    # Find the rows where the max index is less than n
    # logits = logits[np.argmax(logits, axis=1) < 398]


    print(embeddings.shape)
    print(logits.shape)

    ## PCA of embeddings

    # Perform PCA
    n = 80

    pca = PCA(n_components = n)
    pca.fit(embeddings)

    # Calculate the cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot the cumulative explained variance
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
    plt.title('Cumulative Variance Explained by Principal Components')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Variance Explained')
    plt.grid(True)
    plt.show()


    # Get the first 80 component scores
    num_components = n
    component_scores = pca.transform(embeddings)[:, :num_components]
    print(component_scores.shape)

    # Initialize the matrix to store regression coefficients
    coeff_matrix = np.zeros((num_components, logits.shape[1]))

    # Iterate through each component and perform linear regression
    for i in range(num_components):
        # Extract the i-th principal component
        component_i = component_scores[:, i].reshape(-1, 1)

        # Fit linear regression model
        model = LinearRegression()
        model.fit(component_i, logits)

        # Save regression coefficient in the matrix
        coeff_matrix[i, :] = model.coef_.ravel()

    # Find the maximum value in each row
    max_values_per_row = np.max(np.abs(coeff_matrix), axis=1)

    # Sort the maximum values
    sorted_max_values = np.sort(max_values_per_row)

    # Plot the sorted maximum values
    plt.plot(max_values_per_row, marker='o', linestyle='-', color='b')
    plt.title('Maximum Values per Row')
    plt.xlabel('Row')
    plt.ylabel('Maximum Value')
    plt.grid(True)
    plt.show()


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

    x_labels = list(label_dict.values())[:50]
    x_labels[:] = (elem.split(',')[0] for elem in x_labels)

    # Plot the heatmap
    plt.figure(figsize=(10, 6), layout="tight")
    plt.imshow(coeff_matrix[:,:50], cmap='viridis', aspect='auto', interpolation='none')

    plt.xticks(np.arange(len(x_labels)), labels=x_labels, rotation=90)

    # Add labels and title
    plt.colorbar(label='Regression Coefficient')
    plt.xlabel('Feature Index')
    plt.ylabel('Principal Components')
    plt.title('Regression Coefficients Heatmap')

    plt.show()


    cov = np.cov(component_scores.T)
    plt.figure(figsize=(10, 6), layout="tight")
    plt.imshow(cov, cmap='viridis', aspect='auto', interpolation='none')
    plt.show()

    np.savetxt("covariance_matrix.txt", cov)

    pk.dump(pca, open("pca.pkl","wb"))


    ## Generate some images

    # cov = np.cov(component_scores.T)
    # print(cov.shape)
    # mean = np.mean(component_scores, axis=0)
    # print(mean.shape)
    # random_pcs = np.random.multivariate_normal(mean, cov, size=10)
    # # random_pcs = np.random.standard_normal(size=((10,80)))
    # print(random_pcs.shape)

    # random_embeds = pca.inverse_transform(random_pcs)
    # # random_embeds = np.dot(random_pcs, pca.components_) + pca.mean_
    # print(random_embeds.shape)

    # pipe = unCLIP_evolution.prep_model('../stable-diffusion-2-1-unclip-small')

    # for i in range(10):
    #     unCLIP_evolution.generate_image(pipe, random_embeds[i,:], f"generation/random-img-{i}.png")


    max_indices = np.argmax(logits, axis=1)
    # Create a histogram of the max indices
    plt.hist(max_indices, bins=398, edgecolor='black')
    
    # Customize the plot
    plt.title('Histogram of Max Indices in Each Row')
    plt.xlabel('Index')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)

    # Show the plot
    plt.show()

    # Create a UMAP model
    umap_model = umap.UMAP(n_neighbors=5, min_dist=0.3, metric='euclidean')

    # Fit the UMAP model to your data
    umap_result = umap_model.fit_transform(component_scores)

    # max_indices = np.arange(3932)

    # Plot the UMAP result
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=max_indices)
    plt.title('UMAP Projection')
    plt.colorbar()
    plt.show()

    pk.dump(umap_model, open("umap.pkl","wb"))


    # # Isomap model
    # isomap_model = Isomap(n_neighbors=5, n_components=2)

    # # Fit the Isomap model to your data
    # isomap_result = isomap_model.fit_transform(component_scores)

    # # Plot the Isomap result
    # plt.scatter(isomap_result[:, 0], isomap_result[:, 1], c=max_indices)

    # # Show the plot
    # plt.title('ISOMAP Projection')
    # plt.colorbar()
    # plt.show()