import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def recombination(x, err, mutation_rate, mutation_size, top_n):
    # Classic recombination algorithm
    # Get indexes of top values
    min_err = np.argsort(err)[:top_n]

    # Create next generation
    pop_size = x.shape[0]
    vec_size = x.shape[1]

    top_x = np.zeros((top_n, vec_size))
    for i in range(top_n):
        top_x[i, :] = x[min_err[i], :]

    next_x = np.zeros((pop_size, vec_size))
    for i in range(pop_size):
        parent_idx = np.random.choice(top_n, replace=False, size=2)
        parent_1 = top_x[parent_idx[0]]
        parent_2 = top_x[parent_idx[1]]
        child = np.zeros((1, vec_size))

        for j in range(vec_size):
            choice = np.random.random()
            if choice <= 0.5:
                child[0, j] = parent_1[j]
            else:
                child[0, j] = parent_2[j]
        next_x[i, :] = child

    # next_x[pop_size-1, :] = np.random.randn(1, vec_size)

    # Apply mutations

    # Generate random vecs to add
    y = np.random.uniform(-mutation_size, mutation_size,
                          size=(pop_size, vec_size))
    # Create mask array
    z = np.random.binomial(1, mutation_rate, size=(pop_size, vec_size))
    # Create mutation vec
    mutate = np.multiply(y, z)

    # Add to pop
    next_x = np.add(next_x, mutate)

    return next_x


def plot_cma(x_out_tot):
    # Plot CMA on 2D PCA plot

    data = x_out_tot.reshape(
        x_out_total.shape[0]*x_out_total.shape[1], x_out_total.shape[2])

    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data)

    # The transformed data contains the principal components
    print("Original data shape:", data.shape)
    print("Transformed data shape:", principal_components.shape)
    # print("Principal Components:\n", principal_components)

    group_index = np.arange(len(data)) // 10

    # Plot the scores on a scatter plot with different colors for each observation
    plt.scatter(
        principal_components[:, 0], principal_components[:, 1], c=group_index, cmap='viridis')

    # Add labels and a colorbar
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Observation Index')

    # Show the plot
    plt.show()

    return None


def cma(x, err, x_out, cov_mats, iter):
    # Run CMA strategy
    # Sort solutions
    # Get top pop_size/2 vectors
    pop_size = x.shape[0]
    top_n = int(pop_size/2)
    vec_size = x.shape[1]

    min_err_idx = np.argsort(err)[:top_n]
    min_err = np.zeros(min_err_idx.shape)

    top_x = np.zeros((top_n, vec_size))
    for i in range(top_n):
        top_x[i, :] = x[min_err_idx[i], :]
        min_err[i] = err[min_err_idx[i]]

    # Get mean and covariance matrix

    # Mean: Weighted sum of top_x based on error values
    weights = np.divide(np.ones(min_err.shape), min_err)
    weights_normalized = weights/weights.sum()
    # print(weights_normalized)
    # print(weights_normalized.sum())

    mean = np.sum((top_x.T*weights_normalized).T, axis=0)
    # mean = np.mean(top_x, axis=0)

    # cov_mat = None
    # if np.sum(x_out) > 0:
    #     cov_mat = np.cov(x_out.T)
    # else:
    #     cov_mat = np.cov(x.T)

    cov_mat = np.eye(vec_size)

    # this_cov_mat = np.cov(x.T)
    # cov_mats[iter, :, :] = this_cov_mat
    # cov_mat = np.mean(cov_mats[:(iter+1), :, :], axis=0)

    # cov_mat = np.cov(x.T)

    sigma = np.ones(vec_size) * 0.3

    next_x = np.random.multivariate_normal(
        mean=mean, cov=(sigma**2) * cov_mat, size=pop_size)

    # # Generate random vecs to add
    # mutation_size = 0.25
    # mutation_rate = 0.02
    # y = np.random.uniform(-mutation_size, mutation_size,
    #                       size=(pop_size, vec_size))
    # # Create mask array
    # z = np.random.binomial(1, mutation_rate, size=(pop_size, vec_size))
    # # Create mutation vec
    # mutate = np.multiply(y, z)
    # # Add to pop
    # next_x = np.add(next_x, mutate)

    return next_x, cov_mats


def custom(x, err, mutation_rate, mutation_size):

    # Create next generation
    pop_size = x.shape[0]
    vec_size = x.shape[1]

    top_n = int(pop_size/2)
    min_err = np.argsort(err)[:top_n]

    top_x = np.zeros((top_n, vec_size))
    for i in range(top_n):
        top_x[i, :] = x[min_err[i], :]

    next_x = np.zeros((pop_size, vec_size))

    weights = err[min_err][:top_n] / np.sum(err[min_err][:top_n])
    mean = np.sum((top_x.T*weights).T, axis=0)

    for i in range(pop_size):
        next_x[i, :] = mean
        y = np.random.randn(1, vec_size) * mutation_size
        z = np.random.binomial(1, mutation_rate, size=(1, vec_size))
        mutate = np.multiply(y, z)
        next_x[i, :] = next_x[i, :] + mutate

    # Add to pop
    next_x = np.add(next_x, mutate)

    return next_x


def run_evolution(
    target,
    pop_size=10,
    vec_size=1024,
    max_iters=300,
    top_n=3,
    mutation_size=1,
    mutation_rate=0.02,
    error_power=2,
    mode="classic"
):

    # Initialize outputs
    err_out = np.zeros((1, max_iters))  # Minimum error per iteration
    x_out = np.zeros((max_iters, vec_size))  # Lowest-error vector
    x_out_total = np.zeros((max_iters, pop_size, vec_size))  # All x vector

    # print(target)
    # print(np.random.permutation(target)
    # Create initial pop
    # x = np.random.randn(pop_size, vec_size)
    x = np.zeros((pop_size, vec_size))
    for i in range(pop_size):
        x[i, :] = target

    x = np.random.permutation(
        x.reshape(pop_size*vec_size)).reshape((pop_size, vec_size))
    print(x)

    # Stare vars for CMA
    cov_mats = np.zeros((max_iters, vec_size, vec_size))

    # Evolution loop
    for iter in range(max_iters):

        # Create correlation array
        err = np.zeros(pop_size)

        # Test each x against target
        for i in range(pop_size):
            # Get sum of squared error
            err_vec = (target - x[i, :]) ** error_power
            euc_dist = np.sqrt(np.sum(err_vec))
            err[i] = euc_dist

        # Run propagation strategy
        if mode == "classic":
            x = recombination(x, err, mutation_rate, mutation_size, top_n)
        elif mode == "cma":
            x, cov_mats = cma(x, err, x_out, cov_mats, iter)
        elif mode == "custom":
            x = custom(x, err, mutation_rate, mutation_size)
        else:
            raise Exception("ERROR: Mode is not valid!")

        # Print error values every 50 iters
        if iter % 10 == 0:
            print(f"Iteration {iter} - Minimum error: {np.min(err)}")

        err_out[0, iter] = np.min(err)
        x_out[iter, :] = x[np.argmin(err), :]
        x_out_total[iter, :, :] = x

    return err_out, x_out, x_out_total


if __name__ == "__main__":

    # For testing
    # Create target vector

    # Set parameters
    pop_size = 10
    vec_size = 768
    max_iters = 50
    top_n = 3
    mutation_size = 0.3
    error_power = 2  # multiple of 2
    # mutation_rates = [0.005, 0.01, 0.02, 0.05]
    mutation_rate = 0.02

    target = np.reshape(np.loadtxt("embed.txt", dtype="float16"), (1, 768))
    # target = np.random.randn(1, vec_size)
    # a = np.random.choice([-8, -4, 0, 4, 8], size=(1,vec_size), p=[0.0025, 0.0025, 0.99, 0.0025, 0.0025])
    # target = target + a
    # plt.hist(target[0,:], bins=20, color='blue', edgecolor='black')

    # err_out_tot = np.zeros((np.size(mutation_rates), max_iters))
    # x_out_tot = np.zeros((np.size(mutation_rates), vec_size))

    # for mut in range(np.size(mutation_rates)):
    #     mutation_rate = mutation_rates[mut]

    err_out, x_out, x_out_total = run_evolution(
        target=target,
        top_n=top_n,
        pop_size=pop_size,
        mutation_rate=mutation_rate,
        mutation_size=mutation_size,
        max_iters=max_iters,
        error_power=error_power,
        vec_size=vec_size,
        mode="custom")

    plot_cma(x_out_total)

    # err_out_tot[mut, :] = err_out
    # x_out_tot[mut, :] = x_out

    plt.plot(np.arange(max_iters), err_out[0, :], 'orange', label='0.02')
    plt.show()

    plt.scatter(target[0, :], x_out[max_iters-1, :])
    # obtain m (slope) and b(intercept) of linear regression line
    m, b = np.polyfit(target[0, :], x_out[max_iters-1, :], 1)

    # add linear regression line to scatterplot
    plt.plot(target[0, :], m*target[0, :]+b)
    plt.xlim(-3, 3)
    plt.ylim(-3, 3)

    # If last iter, save vec
    # final_vec = x_out_tot[3,:]
    # np.savetxt('final.txt', final_vec, delimiter=',')

    # Plot a simple line chart
    # plt.plot(np.arange(max_iters), err_out_tot[0, :], 'orange', label='0.005')
    # plt.plot(np.arange(max_iters), err_out_tot[1, :], 'green', label='0.01')
    # plt.plot(np.arange(max_iters), err_out_tot[2, :], 'blue', label='0.02')
    # plt.plot(np.arange(max_iters), err_out_tot[3, :], 'red', label='0.05')

    # plt.xlabel('Iteration')
    # plt.ylabel('Error')

    # plt.plot(np.arange(vec_size), target[0, :], 'yellow', label='target')
    # plt.plot(np.arange(vec_size), x_out_tot[0, :], 'orange', label='0.005')
    # plt.plot(np.arange(vec_size), x_out_tot[1, :], 'green', label='0.01')
    # plt.plot(np.arange(vec_size), x_out_tot[2, :], 'blue', label='0.02')
    # plt.plot(np.arange(vec_size), x_out_tot[3, :], 'red', label='0.05')

    # plt.scatter(target, x_out_tot[3, :])

    plt.legend()
    plt.show()

    cov = np.cov(x_out_total.reshape(max_iters*x_out_total.shape[1], vec_size))
    plt.figure(figsize=(10, 8))
    heatmap = plt.imshow(cov, cmap='coolwarm',
                         interpolation='nearest')
    plt.colorbar(heatmap, label='Covariance')
    plt.title('Covariance Matrix Heatmap')
    plt.show()
