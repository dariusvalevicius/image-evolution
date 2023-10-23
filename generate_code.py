import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def run_evolution(
    target,
    pop_size = 10,
    vec_size = 768,
    max_iters = 300,
    top_n = 3,
    mutation_size = 1,
    mutation_rate = 0.02,
    error_power = 2
    ):

    # Initialize outputs
    err_out = np.zeros((1, max_iters)) # Minimum error per iteration
    x_out = np.zeros((1, vec_size)) # Lowest-error vector
    
    # Create initial pop    
    x = np.random.randn(pop_size, vec_size)
    #print(x)


    # Evolution loop
    for iter in range(max_iters):

        # Create correlation array
        err = np.zeros(pop_size)

        # Test each x against target
        for i in range(pop_size):
            # Get sum of squared error
            err_vec = (target - x[i,:]) ** error_power
            euc_dist = np.sqrt(np.sum(err_vec))
            err[i] = euc_dist
        # Test each x against target
        # for i in range(pop_size):
            # Get cosine similarity
            # cos_sim = np.dot(target, x[i,:])/(np.linalg.norm(target)*np.linalg.norm(x[i,:]))
            # Get dot product
            # dot = np.dot(target, x[i,:])
            # err[i] = -dot

        # Get indexes of top values
        # err
        min_err = np.argsort(err)[:top_n]

        # If last iter, save vec
        if iter == max_iters-1:
            final_vec = x[min_err[0],:]
            np.savetxt('final.txt', final_vec, delimiter=',')

        # Create next generation
        top_x = np.zeros((top_n, vec_size))
        for i in range(top_n):
            top_x[i,:] = x[min_err[i],:]
        
        next_x = np.zeros((pop_size, vec_size))
        for i in range(pop_size-1):
            parent_idx = np.random.choice(top_n, replace=False, size=2)
            parent_1 = top_x[parent_idx[0]]
            parent_2 = top_x[parent_idx[1]]
            child = np.zeros((1, vec_size))

            for j in range(vec_size):
                choice = np.random.random()
                if choice <= 0.5:
                    child[0,j] = parent_1[j]
                else:
                    child[0,j] = parent_2[j]
            next_x[i,:] = child
        
        next_x[pop_size-1,:] = np.random.randn(1, vec_size)

        # Apply mutations

        # Generate random vecs to add
        y = np.random.uniform(-mutation_size, mutation_size, size=(pop_size, vec_size))
        # y = np.random.uniform(mutation_size/2, mutation_size + (mutation_size/2), size=(pop_size, vec_size))
        # y2 = np.random.choice([-1,1], size = (pop_size,vec_size))
        # y = np.multiply(y,y2)
        # y = np.random.randn(pop_size, vec_size) * mutation_size
        # y = np.random.standard_cauchy(size=(pop_size, vec_size)) * mutation_size
        # y[(y>-8) & (y<8)] = 0

        # y = np.random.choice([-mutation_size, mutation_size], size=(pop_size,vec_size))

        # Create mask array
        z = np.random.binomial(1, mutation_rate, size=(pop_size, vec_size))

        # Create mutation vec
        mutate = np.multiply(y, z)
        # print(f"Sum of mutate: {np.sum(mutate)}")

        # Add to pop
        next_x = np.add(next_x, mutate)

        # Set next gen
        x = next_x

        if iter % 50 == 0:
            print(f"Iteration {iter} - Minimum error: {np.min(err)}")

        err_out[0, iter] = np.min(err)
        x_out[0, :] = x[np.argmin(err), :]

    return err_out, x_out


if __name__ == "__main__":

    # For testing
    # Create target vector

    # Set parameters
    pop_size = 10
    vec_size = 768
    max_iters = 150
    top_n = 3
    mutation_size = 1
    error_power = 4 # multiple of 2
    # mutation_rates = [0.005, 0.01, 0.02, 0.05]
    mutation_rates = [0.02]



    target = np.reshape(np.loadtxt("embed.txt", dtype="float16"), (1,768))
    # target = np.random.randn(1, vec_size)
    # a = np.random.choice([-8, -4, 0, 4, 8], size=(1,vec_size), p=[0.0025, 0.0025, 0.99, 0.0025, 0.0025])
    # target = target + a
    # plt.hist(target[0,:], bins=20, color='blue', edgecolor='black')


    err_out_tot = np.zeros((np.size(mutation_rates), max_iters))
    x_out_tot = np.zeros((np.size(mutation_rates), vec_size))

    for mut in range(np.size(mutation_rates)):
        mutation_rate = mutation_rates[mut]


        err_out, x_out = run_evolution(
            target=target, 
            mutation_rate=mutation_rate, 
            mutation_size=1, 
            max_iters=max_iters, 
            error_power=6)

        err_out_tot[mut, :] = err_out
        x_out_tot[mut, :] = x_out
        


    # If last iter, save vec
    # final_vec = x_out_tot[3,:]
    # np.savetxt('final.txt', final_vec, delimiter=',')

    # Plot a simple line chart
    plt.plot(np.arange(max_iters), err_out_tot[0, :], 'orange', label='0.005')
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

        


    

        



