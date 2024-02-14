import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import pickle
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree


global pop_size
pop_size = 9
global stim_time
stim_time = 5
global washout_time
washout_time = 1




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
    data1 = np.loadtxt(file1, delimiter=",")[:,1:]

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

    data2 = np.loadtxt(file2, delimiter=",")[:,1:]

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
    data = np.loadtxt(file, delimiter=",")[:,1:]

    top_image_indices = np.argmax(data, axis=1)

    directory = os.path.dirname(file)
    # Assuming you have a list of image paths (replace this with your own paths)
    # Construct file names based on indices
    image_paths = []
    for i in range(len(top_image_indices)):
        image_paths.append(f"{directory}/images/generation_{i}/num-{top_image_indices[i]+1}.PNG")

    # Create a figure and axes for each image
    fig, axes = plt.subplots(1, len(image_paths), figsize=(len(image_paths), 2))

    # Iterate through image paths and plot each image with a number
    for i, image_path in enumerate(image_paths):
        # Load and plot the image
        img = plt.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')  # Hide axes for better visualization
        axes[i].set_title(f'{data[i, top_image_indices[i]]:.2f}', loc="left")

    base_name = os.path.basename(file)  # Get the last component of the path
    filename, extension = os.path.splitext(base_name)

    fig.suptitle(f"Top images from each generation: {filename}")
    # Adjust layout to prevent overlap
    plt.tight_layout()


def correlation_plot(file1, file2, condition):
    data1 = np.loadtxt(file1, delimiter=",")[:,1:]
    data1 = data1.reshape(data1.shape[0]*data1.shape[1])
    data2 = np.loadtxt(file2, delimiter=",")[:,1:]
    data2 = data2.reshape(data2.shape[0]*data2.shape[1])

    # Create a DataFrame
    data = {'rating': data1, 'scr': data2}
    df = pd.DataFrame(data)
    # df = df[df['scr'] > 0]

    # Fit a linear regression model
    model = sm.OLS(df['scr'], sm.add_constant(df['rating'])).fit()

    # Plotting the data with regression line
    sns.regplot(x='rating', y='scr', data=df)#, scatter_kws={"s": 80})

    # Adding labels and title
    plt.xlabel('Rating')
    plt.ylabel('SCR')
    plt.title(condition)

    # Annotate plot with R-value and significance
    r_value = model.rsquared**0.5
    p_value = model.pvalues['rating']

    # Format the annotation text
    annotation_text = f'R-value: {r_value:.2f}\nP-value: {p_value:.4f}'

    # Add the annotation to the plot
    plt.annotate(annotation_text, xy=(0.3, 0.8), xycoords='axes fraction', fontsize=10, ha='center')

    # Print the regression summary
    print(model.summary())


def plot_eda(file1, file2):

    data = np.loadtxt(file1, delimiter=",")
    data = np.delete(data, np.arange(0, data.shape[0], pop_size), axis=0)
    scr_scores = np.loadtxt(file2, delimiter=",")[:,1:].flatten()

    data_centered = data
    n_lines, n_points = data.shape

    for i in range(n_lines):
        data_centered[i,:] = (data[i,:] - data[i,0])

    # Generate random colors with slight variation
    colors_resp = plt.cm.autumn(np.linspace(0, 0.5, n_lines))
    # colors_in_between = plt.cm.viridis(np.linspace(0.4, 0.6, n_lines))
    colors_no_resp = plt.cm.winter(np.linspace(0, 0.5, n_lines))

    num_responses = 0

    # Generate data for each line
    for i in range(n_lines):    


        # if data[i,20:20].max() - data[i,5:10].mean() > 0.01:
        if scr_scores[i] > 0:
            colors = colors_resp
            num_responses = num_responses + 1
            # plt.plot(data_centered[i,:25], color=colors[i], alpha=0.25)

        # elif data[i,30:70].mean() - data[i,:20].mean() > 0.01:
        #     colors = "orange"
        else:
            colors = colors_no_resp
            # plt.plot(data_centered[i,:25], color=colors[i], alpha=0.25)



        # Plot the line with varying color
        plt.plot(data_centered[i,:25], color=colors[i], alpha=0.25)

    plt.vlines(4, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='black')
    plt.vlines(19, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='black')
    # plt.vlines(15, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='red')
    # plt.vlines(30, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='red')

    # Annotate the graph with a text label
    plt.annotate(f'Num responses:\n{num_responses}/{n_lines}', xy=(0, 0), xytext=(0, data.max()/2))


def plot_scr_by_trial(file):
    data = np.loadtxt(file, delimiter=",")

    m, n = data.shape

    # Calculate means and standard deviations (for error bars)
    means = np.mean(data, axis=0)
    std_devs = np.std(data, axis=0)

    # Plot the raw data points
    plt.plot(data.T, 'o', alpha=0.5, color="black")

    # Plot the means with error bars
    plt.errorbar(np.arange(n), means, yerr=std_devs, fmt='o-', color='black')

    plt.xticks(np.arange(n), (np.arange(n)+1).astype(int))
    # plt.yticks(np.arange(11))

    # Customize the plot
    plt.xlabel('Trial')
    plt.ylabel('Values')
    plt.title('Scores by trial within generation')

def plot_umap(participant):

    data = np.empty((0, 80))
    scr_scores = np.empty((0, pop_size-1))
    ratings = np.empty((0, pop_size-1))

    for condition in ["selfreport", "scr", "control"]:
        root = f"data/{participant}/{condition}"   

        # folder = f"{root}/figures/"
        # if not os.path.exists(folder):
        #     os.mkdir(folder)

        data_new = np.loadtxt(f"{root}/embeddings.txt", delimiter=",")
        data_new = np.delete(data_new, np.arange(0, data_new.shape[0], pop_size), axis=0)
        print(data_new.shape)

        data = np.vstack((data, data_new))
        print(data.shape)
        scr_scores = np.vstack((scr_scores, np.loadtxt(f"{root}/scr_scores.txt", delimiter=",")[:,1:]))
        print(scr_scores.shape)
        ratings = np.vstack((ratings, np.loadtxt(f"{root}/ratings.txt", delimiter=",")[:,1:]))

    scr_scores = scr_scores.flatten()
    ratings = ratings.flatten()

    # Load the UMAP model from the pickle file
    with open('umap.pkl', 'rb') as file:
        umap_model = pickle.load(file)

    # Transform your data using the UMAP model
    umap_transformed = umap_model.transform(data)

    # Plot the transformed vectors in the UMAP space
    plt.scatter(umap_transformed[:, 0], umap_transformed[:, 1], c=ratings, cmap='viridis', alpha=0.6)
    plt.title('UMAP Visualization: Ratings')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(label='Rating')
    plt.show()

    # Plot the transformed vectors in the UMAP space
    plt.scatter(umap_transformed[:, 0], umap_transformed[:, 1], c=scr_scores, cmap='viridis', alpha=0.6)
    plt.title('UMAP Visualization: SCR')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.colorbar(label='SCR Peak')
    plt.show()

    x = umap_transformed[:, 0]
    y = umap_transformed[:, 1]
    magnitudes = ratings

    points = np.column_stack((x, y))
    grid_size = 40

    # Create an evenly spaced grid over the XY plane
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_grid = np.linspace(-20, 20, grid_size)
    y_grid = np.linspace(-20, 20, grid_size)
     # Initialize arrays to store grid points and their corresponding magnitudes
    grid_points = np.zeros((grid_size, grid_size, 2))
    grid_magnitudes = np.zeros((grid_size, grid_size))

    # Iterate over each point and assign it to the corresponding grid cell
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the boundaries of the current grid cell
            x_min_cell = x_grid[i]
            x_max_cell = x_grid[i+1] if i < grid_size - 1 else x_max
            y_min_cell = y_grid[j]
            y_max_cell = y_grid[j+1] if j < grid_size - 1 else y_max

            # Find points within the current grid cell
            mask = np.logical_and.reduce((
                points[:, 0] >= x_min_cell,
                points[:, 0] <= x_max_cell,
                points[:, 1] >= y_min_cell,
                points[:, 1] <= y_max_cell
            ))
            points_in_cell = points[mask]
            magnitudes_in_cell = magnitudes[mask]

            # Compute average magnitude for points in the cell
            if len(points_in_cell) > 0:
                average_magnitude = np.mean(magnitudes_in_cell)
            else:
                average_magnitude = 0 #np.nan  # No points in the cell

            # Store grid point and average magnitude
            grid_points[i, j] = [(x_min_cell + x_max_cell) / 2, (y_min_cell + y_max_cell) / 2]
            grid_magnitudes[i, j] = average_magnitude

    grid_points = grid_points.reshape(-1, 2)
    averages = grid_magnitudes.ravel()


    # Calculate KDE with magnitudes as weights
    kde = gaussian_kde([grid_points[:,0], grid_points[:,1]], weights=averages)

    # Define grid
    # x_grid = np.linspace(-20, 20, 100)
    # y_grid = np.linspace(-20, 20, 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = kde.evaluate([X.ravel(), Y.ravel()]).reshape(X.shape)

    # Plot KDE
    plt.figure(figsize=(8, 6))
    sns.heatmap(Z, cmap='viridis', alpha=0.8)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('2D Kernel Density Estimation with Magnitudes')
    plt.show()

    # Plot KDE in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    # # Set labels and title
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Density')
    # ax.set_title('3D Kernel Density Estimation with Magnitudes')

    # plt.show()





def create_all(participant, condition, show, save):

    root = f"data/{participant}/{condition}"   

    folder = f"{root}/figures/"
    if not os.path.exists(folder):
        os.mkdir(folder)

    correlation_plot(f"{root}/ratings.txt", f"{root}/scr_scores.txt", condition)
    if save: plt.savefig(f"{root}/figures/correlation.png")
    plt.show() if show else plt.clf()

    plot_top_images(f"{root}/ratings.txt")
    if save: plt.savefig(f"{root}/figures/top_images_ratings.png")
    plt.show() if show else plt.clf()

    plot_top_images(f"{root}/scr_scores.txt")
    if save: plt.savefig(f"{root}/figures/top_images_scr.png")
    plt.show() if show else plt.clf()

    plot_both(f"{root}/ratings.txt", f"{root}/scr_scores.txt")
    plt.title(f'Image evolution: {condition} using scr_scores')
    if save: plt.savefig(f"{root}/figures/scores_fig.png")
    plt.show() if show else plt.clf()

    plot_both(f"{root}/ratings.txt", f"{root}/scr_means.txt")
    plt.title(f'Image evolution: {condition} using scr_means')
    if save: plt.savefig(f"{root}/figures/means_fig.png")
    plt.show() if show else plt.clf()

    plot_eda(f"{root}/scr_data.txt", f"{root}/scr_scores.txt")
    plt.title('EDA data')
    if save: plt.savefig(f'{root}/figures/eda_data.png')
    plt.show() if show else plt.clf()

    plot_scr_by_trial(f"{root}/scr_scores.txt")
    if save: plt.savefig(f'{root}/figures/scr_by_trial.png')
    plt.show() if show else plt.clf()





if __name__ == "__main__":

    # create_all(50, "selfreport", show=True, save=False)
    # create_all(51, "scr", show=True, save=False)
    # create_all(52, "control", show=True, save=False)
    # create_all(675972, "scr", show=True, save=False)
    # create_all(69630, "scr", show=True, save=False)
    # create_all(810175, "scr", show=True, save=False)
    # create_all(701338, "scr", show=True, save=False)
    # create_all(8193, "scr", show=True, save=True)



    plot_umap(50)
    # if save: plt.savefig(f'{root}/figures/umap.png')
    plt.show()