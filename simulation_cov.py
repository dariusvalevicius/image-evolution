import unCLIP_evolution
import classify
import shutil
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pickle as pk

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

    # Model paths
    unclip_model_path = "../stable-diffusion-2-1-unclip-small"
    vit_model_path = "../vit-base-patch16-224"

    # Define parameters

    pop_size = 12  # Make this a multiple of 4
    top_n = int(pop_size/2)
    mutation_size = 0.3
    vec_size = 80
    embedding_size = 768
    max_iters = 15
    imagenet_index = 2
    diffusion_steps = 15

    # Load PCA and covariance matrix
    pca = pk.load(open("pca.pkl",'rb')) 
    umap = pk.load(open("umap.pkl",'rb'))
    cov = np.loadtxt("covariance_matrix.txt")

    # Start with random embeddings
    # x = np.random.randn(pop_size, vec_size)
    x = np.random.multivariate_normal(np.zeros(vec_size), cov, size=pop_size)
    # print(f"X: {x}:")

    # For testing: random target vector
    target = np.random.randn(vec_size).reshape(1, vec_size)
    # target = np.reshape(np.loadtxt("embed.txt", dtype="float16"), (1, 768))
    # print(f"T: {target}")

    # Prepare models
    pipe = unCLIP_evolution.prep_model(unclip_model_path)
    vit_processor, vit_model = classify.prep_model(vit_model_path)

    # Data to save
    all_x = np.zeros((max_iters, pop_size, vec_size))
    best_x = np.zeros((max_iters, vec_size))
    best_fitness = np.zeros(max_iters)

    means = np.zeros((max_iters, vec_size))
    sigmas = np.zeros(max_iters)

    mutation_rate_selected = np.zeros(pop_size)
    all_rates = np.zeros(max_iters)

    # Loop
    for iter in range(max_iters):

        fitness = np.zeros(pop_size)

        # Test fitness of each vector
        for j in range(pop_size):

            # Generate the images
            image_name = f"generation/img_{j}.png"


            pcs = x[j, :]
            # Convert pca space to CLIP space
            embedding = pca.inverse_transform(pcs)

            unCLIP_evolution.generate_image(
                pipe, embedding, image_name, diffusion_steps)

            # Evaluate the image
            fitness[j] = classify.return_score(
                vit_processor, vit_model, image_name)[0, imagenet_index]

            # For testing: euclidian distance metric
            # fitness[j] = 1 / np.linalg.norm(target - x[j, :])

        # print(fitness)

        # Get top vectors
        idx = np.argsort(fitness)[::-1]
        fitness_sorted = fitness[idx]
        x_sorted = x[idx, :]

        fitness_top = fitness_sorted[:top_n]
        x_top = x_sorted[:top_n, :]

        # Update output vars
        best_x[iter, :] = x_top[0, :]
        best_fitness[iter] = fitness_top[0]

        # Every 5 generations: Save figure
        print(f"Iteration: {iter + 1}, Fitness: {fitness_top[0]}, Sigma: {mutation_size}")
        shutil.copy(f"generation/img_{idx[0]}.png",
                        f"saved_images/iter_{iter}.png")

        # Compute recombination probability weights
        median = np.median(fitness)
        fitness_relative = np.clip(fitness_top - median, 0, None)
        weights = fitness_relative / np.sum(fitness_relative)
        # print(weights)

        # Produce next generation

        # next_x = np.zeros(x.shape)

        ## CMA test

        # mean = np.sum(x, axis=0)
        # # print(mean.shape)

        # x_top_unique = x_top - mean
        # # print(x_top_unique.shape)

        # x_top_weighted = (x_top_unique.T * weights).T
        # mean = mean + np.sum(x_top_weighted, axis=0)

        mean = np.sum((x_top.T * weights).T, axis=0)
        next_x = np.random.multivariate_normal(mean, mutation_size * np.identity(vec_size), size=pop_size)
        next_x = np.random.multivariate_normal(mean, mutation_size * cov, size=pop_size)


        # if iter == (max_iters - 1):
        #     image_name = f"generation/img_final.png"

        #     pcs = mean
        #     # Convert pca space to CLIP space
        #     embedding = pca.inverse_transform(pcs)

        #     unCLIP_evolution.generate_image(
        #         pipe, embedding, image_name)


        # for j in range(pop_size):
        #     for k in range(vec_size):
        #         # Random number
        #         rand = np.random.uniform()
        #         # Select gene base on roll
        #         for l in range(top_n):
        #             if rand <= np.sum(weights[:l+1]):
        #                 next_x[j, k] = x_top[l, k]
        #                 # print(f"Choosing: {l}")
        #                 break

        # # Add mutations
        # mutations = np.zeros(x.shape)
        # for j in range(pop_size):
        #     mutation_rate = mutation_rates[np.random.randint(0, len(mutation_rates))]
        #     y = np.random.randn(vec_size) * mutation_size
        #     z = np.random.binomial(1, mutation_rate, size=vec_size)
        #     mutations[j, :] = np.multiply(y, z)

        #     mutation_rate_selected[j] = mutation_rate

        # nonzero_idx = np.nonzero(mutations)

        # next_x[nonzero_idx] = mutations[nonzero_idx]

        # Update vec
        all_x[iter,:,:] = x
        x = next_x

        # Update sigma
        sigmas[iter] = mutation_size
        means[iter] = mean
        if iter > 1:
            path_2 = means[iter,:] - means[iter-1,:]
            path_1 = means[iter-1,:] - means[iter-2,:]
            cosine = np.dot(path_1, path_2)/(np.linalg.norm(path_1)*np.linalg.norm(path_2))

            mutation_size = mutation_size * (1 + cosine/2)

    # End of loop

    # Plot results
    # plot_pca(np.vstack((best_x, target)), pop_size)
    plot_pca(best_x)
    plt.savefig("figures/pca.png")
    plt.show()
    plt.clf()

    # Fit the UMAP model to your data
    # umap_result = umap.transform(all_x.reshape((max_iters*pop_size, vec_size)))
    umap_result = umap.transform(best_x)

    # Plot the UMAP result
    plt.scatter(umap_result[:, 0], umap_result[:, 1], c=np.arange(len(umap_result)))
    plt.title('UMAP Projection')
    plt.colorbar()
    plt.savefig("figures/umap.png")
    plt.show()

    plt.plot(np.arange(max_iters), best_fitness, 'orange')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.savefig("figures/error.png")
    plt.show()
    plt.clf()

    plt.plot(np.arange(max_iters), sigmas)
    plt.xlabel('Iteration')
    plt.ylabel('Mutation rate')
    plt.savefig("figures/mutation_rates.png")
    plt.show()
    plt.clf()

    # plt.scatter(target[0, :], best_x[-1, :])
    # # obtain m (slope) and b(intercept) of linear regression line
    # m, b = np.polyfit(target[0, :], best_x[-1, :], 1)
    # # add linear regression line to scatterplot
    # plt.plot(target[0, :], m*target[0, :]+b)
    # plt.xlabel('Target vector')
    # plt.ylabel('Last evolved vector')
    # plt.savefig("figures/correlation.png")
    # # plt.show()
    # plt.clf()
