import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

global pop_size
pop_size = 9
global max_iters
max_iters = 10
global stim_time
stim_time = 5

global data_types
data_types = {
        'eda': 10,
        'hr': 5,
        'temp': 5
    }


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

def plot_eda(file1, file2, freq):

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

        if scr_scores[i] > 0:
            colors = colors_resp
            num_responses = num_responses + 1
        else:
            colors = colors_no_resp

        # Plot the line with varying color
        plt.plot(data_centered[i,:freq*5], color=colors[i], alpha=0.25)

    plt.vlines(freq*1, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='black')
    plt.vlines(freq*4, ymin=data.min(), ymax=data.max(), linestyles='dotted', color='black')

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
    
def interpolate_whole_file(data, rate):

    data['time'] = pd.to_datetime(data['time'], format="%H:%M:%S.%f")

    values = data['value']
    time_differences = data['time'] - data['time'].iloc[0]
    timestamps = time_differences.dt.total_seconds().astype(float).values.tolist()

    # Fixed timestep for the curve
    fixed_timestep = 1/rate  # Choose your desired timestep

    # Interpolate data points
    interp_func = np.interp(np.arange(timestamps[0], timestamps[-1]+fixed_timestep, fixed_timestep), timestamps, values)

    return interp_func


def plot_block(participant, condition, data_type):
    df = pd.read_csv(f'data/{participant}/{condition}/{data_type}_data.txt', header=None, names=["value","time"], delimiter=',')
    series = interpolate_whole_file(df, data_types[data_type])
    # print(hr)
    plt.plot(series)
    plt.title(f"Participant {participant}, Condition: {condition}: {data_type}")
    plt.show()

def interpolate_data(data, timings, rate):
    # # Example data with timestamps and values
    # timestamps = [0, 2, 5, 7]  # Example timestamps
    # values = [10, 15, 20, 25]   # Example values

    output = np.zeros((max_iters*pop_size, stim_time*rate))

    data['time'] = pd.to_datetime(data['time'], format="%H:%M:%S.%f")
    timings['start_time'] = pd.to_datetime(timings['start_time'], format="%H:%M:%S.%f")
    timings['end_time'] = pd.to_datetime(timings['end_time'], format="%H:%M:%S.%f")

    for i in range(10):
        for j in range(9):
            # print(f"Doing trial {i*pop_size + j}/{max_iters*pop_size}")
            trial_times = timings[(timings['generation'] == i) & (timings['trial'] == j)]
            if len(trial_times) > 1:
                trial_times = trial_times.tail(1)
            
            start_time = trial_times.iloc[0]['start_time']
            end_time = trial_times.iloc[0]['end_time']

            trial_data = data[(data['time'] >= start_time) & (data['time'] <= end_time)]
            if len(trial_data) == 0:
                trial_data = data[data['time'] <= end_time].tail(1)


            values = trial_data['value']
            time_differences = trial_data['time'] - trial_data['time'].iloc[0]
            timestamps = time_differences.dt.total_seconds().astype(float).values.tolist()
            # print(timestamps)

            # Fixed timestep for the curve
            fixed_timestep = 1/rate  # Choose your desired timestep

            # Interpolate data points
            interp_func = np.interp(np.arange(0, stim_time, fixed_timestep), timestamps, values)


            output[i*pop_size + j, :] = interp_func
            # print(interp_func)
                
    return output



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

    plot_both(f"{root}/ratings.txt", f"{root}/eda_means.txt")
    plt.title(f'Image evolution: {condition} with EDA')
    if save: plt.savefig(f"{root}/figures/means_fig.png")
    plt.show() if show else plt.clf()

    plot_both(f"{root}/ratings.txt", f"{root}/hr_means.txt")
    plt.title(f'Image evolution: {condition} with HR')
    if save: plt.savefig(f"{root}/figures/means_fig.png")
    plt.show() if show else plt.clf()

    # plot_both(f"{root}/ratings.txt", f"{root}/temp_means.txt")
    # plt.title(f'Image evolution: {condition} with TEMP')
    # if save: plt.savefig(f"{root}/figures/means_fig.png")
    # plt.show() if show else plt.clf()

    plot_eda(f"{root}/eda_data_interp.txt", f"{root}/scr_scores.txt", 10)
    plt.title('EDA data')
    if save: plt.savefig(f'{root}/figures/eda_data.png')
    plt.show() if show else plt.clf()

    # plot_eda(f"{root}/hr_data_interp.txt", f"{root}/scr_scores.txt", 5)
    # plt.title('HR data')
    # if save: plt.savefig(f'{root}/figures/eda_data.png')
    # plt.show() if show else plt.clf()

    # plot_eda(f"{root}/temp_data_interp.txt", f"{root}/scr_scores.txt", 5)
    # plt.title('TEMP data')
    # if save: plt.savefig(f'{root}/figures/eda_data.png')
    # plt.show() if show else plt.clf()

    plot_scr_by_trial(f"{root}/scr_scores.txt")
    if save: plt.savefig(f'{root}/figures/scr_by_trial.png')
    plt.show() if show else plt.clf()

def interpolate_all(participant, condition):

    for data_type, rate in data_types.items():
        root = f"data/{participant}/{condition}"
        data = pd.read_csv(f'{root}/{data_type}_data.txt', header=None, names=["value", "time"], delimiter=',')
        timings = pd.read_csv(f'{root}/trial_timings.txt', header=0, delimiter=',')

        interp_data = interpolate_data(data, timings, rate)
        np.savetxt(f'{root}/{data_type}_data_interp.txt', interp_data, delimiter=',')

        means = np.mean(interp_data[:,1*rate:4*rate], axis=1).reshape(max_iters, pop_size)
        np.savetxt(f'{root}/{data_type}_means.txt', means, delimiter=',')


if __name__ == "__main__":

    # interpolate_all(2, "selfreport")
    create_all(2, "scr", show=True, save=False)

    # plot_block(1, "control", "eda")


    # plot_umap(50)
    # # if save: plt.savefig(f'{root}/figures/umap.png')
    # plt.show()










# def plot_score(file):
#     data = np.loadtxt(file, delimiter=",")

#     m, n = data.shape

#     # Calculate means and standard deviations (for error bars)
#     means = np.mean(data, axis=1)
#     std_devs = np.std(data, axis=1)

#     # Plot the raw data points
#     plt.plot(data, 'o', alpha=0.5, color="black")

#     # Plot the means with error bars
#     plt.errorbar(np.arange(m), means, yerr=std_devs, fmt='o-', color='black')

#     plt.xticks(np.arange(m), (np.arange(m)+1).astype(int))
#     plt.yticks(np.arange(11))

#     # Customize the plot
#     plt.xlabel('Generation')
#     plt.ylabel('Values')
#     plt.title('Scores by generation')

#     # plt.show()