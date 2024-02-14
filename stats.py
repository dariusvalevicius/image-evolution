import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import glob as glob
from sklearn.preprocessing import normalize
import pandas as pd
from statsmodels.stats.anova import AnovaRM 
import matplotlib.image as mpimg

# Define parameters
imagenet_indices = {
    "rabbit": 330,
    "cockroach": 314,
    "gecko": 38,
    "spider": 76,
    "chicken": 137,
    "grasshopper": 311,
    "butterfly": 323,
    "bird": 15,
    "peacock": 84,
    "dog": 207,
    "cat": 281,
    "snake": 61,
    "fish": 391,
    "frog": 31,
    "turtle": 37,
    "beetle": 305,
    "ant": 310,
    "bee": 309,
    "guinea pig": 338,
    "sheep": 348,
    "shark": 2,
    "whale": 147
}



def plot_score(data, color, alpha):


    # data = np.loadtxt(file, delimiter=",")
    # data = normalize(data)

    m, n = data.shape

    # Calculate means and standard deviations (for error bars)
    means = np.mean(data, axis=1)
    std_devs = np.std(data, axis=1)

    # Plot the raw data points
    # plt.plot(data, 'o', alpha=0.5, color="black")

    # Plot the means with error bars
    plt.errorbar(np.arange(m), means, yerr=std_devs, fmt='o-', alpha=alpha, color=color)


if __name__ == "__main__":

    m = 10
    n = 22
    scores = glob.glob("sim/sigma_*/logits.txt")
    # Get a list of predefined colors from matplotlib
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    # If n is greater than the number of default colors, cycle through them
    # colors = default_colors * (n // len(default_colors)) + default_colors[:n % len(default_colors)]

    from matplotlib.colors import to_hex

    colors = [to_hex(plt.cm.viridis(i / 19)) for i in range(20)]

    # colors = [
    #     "#F0F0F0", "#E0E0E0", "#D0D0D0", "#C0C0C0",
    #     "#B0B0B0", "#A0A0A0", "#909090", "#808080",
    #     "#707070", "#606060", "#505050", "#404040",
    #     "#303030", "#202020", "#101010", "#080808",
    #     "#060606", "#040404", "#020202", "#000000"
    # ]



    concatenated_array = np.empty((10, 0))

    for i in range(10,20):
        current_array = np.loadtxt(scores[i], delimiter=',')
        plot_score(current_array, colors[i], 0.5)

        concatenated_array = np.concatenate((concatenated_array, current_array), axis=1)

    # plot_score(concatenated_array, "black", 1)

    plt.xticks(np.arange(m), (np.arange(m)+1).astype(int))
    # Customize the plot
    plt.xlabel('Generation')
    plt.ylabel('Values')
    plt.title('Scores by generation')

    plt.show()


    # Need animal, generation, and logit
    # Create an empty list to store the data for the DataFrame
    df_data = []

    # Iterate over subjects and their matrices
    for subject, file in enumerate(scores):
        data = np.loadtxt(file, delimiter=",")

        num_generations, num_scores = data.shape
        
        # Create DataFrame for the subject
        subject_df = pd.DataFrame({
            'animal': [subject] * (num_generations * num_scores),
            'generation': np.repeat(np.arange(1, num_generations + 1), num_scores),
            'logit': data.flatten()
        })
        
        # Append the subject DataFrame to the list
        df_data.append(subject_df)

    # Concatenate all subject DataFrames into a single DataFrame
    result_df = pd.concat(df_data, ignore_index=True)

    # Display the resulting DataFrame
    print(result_df)


    # Fit the repeated measures ANOVA model
    model = ols('logit ~ generation + C(animal) + generation:C(animal)', data=result_df).fit()

    # Perform ANOVA
    anova_table = sm.stats.anova_lm(model, typ=3)

    # Display ANOVA table
    print(anova_table)

    # # Conduct the repeated measures ANOVA 
    # print(AnovaRM(data=result_df, depvar='logit', 
    #           subject='animal', within=['generation']).fit()) 


    top_score_indices = []
    image_paths = []
    top_scores = np.zeros(20)
    i = 0
    # for animal, index in imagenet_indices.items():
    #     current_array = np.loadtxt(f"sim/{animal}/logits.txt", delimiter=',')
    #     top_score_index = np.unravel_index(current_array.argmax(), current_array.shape)
    #     top_score_indices.append(str(top_score_index))
    #     image_paths.append(f"sim/{animal}/generation_{top_score_index[0]}/img_{top_score_index[1]}.PNG")
    #     top_scores[i] = current_array[top_score_index[0], top_score_index[1]]
    #     i = i + 1

    for i in range(20):
        current_array = np.loadtxt(f"sim/sigma_{i}/logits.txt", delimiter=',')
        top_score_index = np.unravel_index(current_array.argmax(), current_array.shape)
        top_score_indices.append(str(top_score_index))
        image_paths.append(f"sim/sigma_{i}/generation_{top_score_index[0]}/img_{top_score_index[1]}.PNG")
        top_scores[i] = current_array[top_score_index[0], top_score_index[1]]
        i = i + 1




    # Create a 2x11 grid
    fig, axes = plt.subplots(2, 10, figsize=(20, 4))

    # Flatten the 2D array of axes to simplify indexing
    axes = axes.flatten()

    # Iterate over each image and annotate
    for i in range(len(image_paths)):
        # animal = list(imagenet_indices.keys())[i]
        animal = f"sigma_{i}"
        # Load and display the image
        img = mpimg.imread(image_paths[i])
        axes[i].imshow(img)
        axes[i].axis('off')  # Turn off axis ticks and labels

        # Annotate with title and numeric value
        axes[i].set_title(f"{animal}\nVal: {top_scores[i]:.2f}, Idx: {top_score_indices[i]}", fontsize=8)

    fig.suptitle("Top Images from Each Sim")

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()

    # Create a 2x11 grid
    fig, axes = plt.subplots(8, 20, figsize=(20, 8))

    # Iterate over each image and annotate
    for i in range(20):
        # animal = f"sigma_{i}"
        for j in range(8):

            # Load and display the image
            img = mpimg.imread(f"sim/sigma_{i}/generation_9/img_{j}.PNG")
            axes[j,i].imshow(img)
            axes[j,i].axis('off')  # Turn off axis ticks and labels

            if j == 0:
                # Annotate with title and numeric value
                axes[j,i].set_title(f"Sig: {(i+1)*0.05:.2f}", fontsize=8)

    fig.suptitle("Last Generation from Each Sim")

    # Adjust layout to prevent overlapping titles
    plt.tight_layout()

    # Show the plot
    plt.show()

