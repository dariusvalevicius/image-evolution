import unCLIP_evolution
import classify
import shutil
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def plot_pca(data, pop_size):
    # Plot CMA on 2D PCA plot

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    # The transformed data contains the principal components
    # print("Original data shape:", data.shape)
    # print("Transformed data shape:", principal_components.shape)
    # print("Principal Components:\n", principal_components)

    group_index = np.arange(len(data)) // pop_size

    # Plot the scores on a scatter plot with different colors for each observation
    plt.scatter(
        principal_components[:, 0], principal_components[:, 1], c=group_index, cmap='viridis')

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
    top_n = int(pop_size/4)
    mutation_rates = [0.01, 0.02, 0.05, 0.1]
    mutation_size = 0.3
    vec_size = 768
    max_iters = 100

    # Start with random embeddings
    x = np.random.randn(pop_size, vec_size)
    # print(f"X: {x}:")

    # For testing: random target vector
    target = np.random.randn(vec_size)
    # print(f"T: {target}")

    # Prepare models
    pipe = unCLIP_evolution.prep_model(unclip_model_path)
    vit_processor, vit_model = classify.prep_model(vit_model_path)

    # Data to save
    best_x = np.zeros((max_iters, vec_size))
    best_fitness = np.zeros(max_iters)

    mutation_rate_selected = np.zeros(pop_size)
    all_rates = np.zeros(max_iters)

    # Loop
    for iter in range(max_iters):

        fitness = np.zeros(pop_size)

        # Test fitness of each vector
        for j in range(pop_size):

            # Generate the images
            image_name = f"generation/img_{j}.png"
            embedding = x[j, :]
            unCLIP_evolution.generate_image(
                pipe, embedding, image_name)

            # Evaluate the image
            fitness[j] = classify.return_score(
                vit_processor, vit_model, image_name)

            # For testing: euclidian distance metric
            # fitness[j] = np.linalg.norm(target - x[j, :])

        # print(fitness)

        # Get top vectors
        idx = np.argsort(fitness)[::-1]
        # idx = np.argsort(fitness)
        fitness_sorted = fitness[idx]
        x_sorted = x[idx, :]

        fitness_top = fitness_sorted[:top_n]
        x_top = x_sorted[:top_n, :]

        # Which mutation rate was selected?
        # print(f"Mutation rate selected: {mutation_rate_selected[idx[0]]}")
        all_rates[iter] = mutation_rate_selected[idx[0]]

        # Update output vars
        best_x[iter, :] = x_top[0, :]
        best_fitness[iter] = fitness_top[0]

        # Every 5 generations: Save figure
        if iter % 5 == 0 or iter + 1 == max_iters:
            print(f"Iteration: {iter}, Fitness: {fitness_top[0]}")
            # shutil.copy("generation/img_{idx[0]}.png",
            #             "saved_images/iter_{iter}.png")

        # Compute recombination probability weights
        weights = fitness_top / np.sum(fitness_top)

        # Produce next generation

        next_x = np.zeros(x.shape)
        for j in range(pop_size):
            for k in range(vec_size):
                # Random number
                rand = np.random.uniform()
                # Select gene base on roll
                for l in range(top_n):
                    if rand <= np.sum(weights[:l+1]):
                        next_x[j, k] = x_top[l, k]
                        # print(f"Choosing: {l}")
                        break

        # Add mutations
        mutations = np.zeros(x.shape)
        for j in range(pop_size):
            mutation_rate = mutation_rates[np.random.randint(0, 4)]
            y = np.random.randn(vec_size) * mutation_size
            z = np.random.binomial(1, mutation_rate, size=vec_size)
            mutations[j, :] = np.multiply(y, z)

            mutation_rate_selected[j] = mutation_rate

        next_x = next_x + mutations

        # Update vec
        x = next_x

    # End of loop

    # Plot results
    plot_pca(best_x, pop_size)
    # plt.show()
    plt.savefig("figures/pca.png")

    plt.plot(np.arange(max_iters), best_fitness, 'orange')
    plt.xlabel('Iteration')
    plt.ylabel('Score')
    plt.savefig("figures/error.png")
    # plt.show()

    plt.plot(np.arange(max_iters), all_rates)
    plt.xlabel('Iteration')
    plt.ylabel('Mutation rate of top vector')
    plt.savefig("figures/mutation_rates.png")
    # plt.show()
