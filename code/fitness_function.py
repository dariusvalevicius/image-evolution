from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import os


def get_scr_scores(root_dir, start_time_dt, onset_times_dt):
    '''
    Function to get SCR scores based on onset times.

    Assumes SCR data is being streamed to {root_dir}/scr_streaming_data.txt,
    with a value column and a timestampe column (formatted %H:%M:%S.%f).

    Returns sum of SCR amplitudes within a time window of [onset, onset + 4s]
    '''

    df = pd.read_csv(os.path.join(root_dir, "scr_streaming_data.txt"), names=["scr", "time"], delimiter="\t")
    df['time'] = pd.to_datetime(df['time'], format='%H:%M:%S.%f')
    print(df.head())

    scr_scores = np.zeros(onset_times_dt.shape[0])

    for i, onset in enumerate(onset_times_dt):

        trial_start = start_time_dt + onset
        trial_end = trial_start + timedelta(seconds=5)

        peak_samp = df[(df['time'] >= trial_start) & (df['time'] <= trial_end)]

        if not peak_samp['scr'].empty:
            scr_scores[i] = peak_samp['scr'].sum()

    return scr_scores


def get_fitness_scores(root_dir, onset_times, start_time, ratings=None, condition=None):
    '''
        This is a modular function that is called at the end of every generation.

        GENERATOR.PY will pass it the stimulus onset times, as well as the subjective ratings if applicable.

        onset_times is the onset times in seconds relative to the initiation of the program (that is, when the participant hit the spacebar after reading the instruction text).

        start_time is passed as a datetime object. Use this to align the stimulus onset times to your recording device, if applicable.
        As such, datetime is a reccomended package for this function.
    '''

    fitness = np.zeros(onset_times.shape[0])
    start_time_dt = datetime.strptime(start_time, '%H:%M:%S.%f')
    onset_times_dt = pd.to_timedelta(onset_times, unit='s')
    pop_size = onset_times.shape[0]

    ######################################################################################################
    # YOUR CODE HERE #
    ######################################################################################################


    if condition == "ratings":
        fitness = ratings
    elif condition == "scr":
        fitness = get_scr_scores(root_dir, start_time_dt, onset_times_dt)
    elif condition == "control":
        fitness = np.random.randn(pop_size)


    ######################################################################################################
    # END #
    ######################################################################################################

    return fitness


if __name__ == "__main__":

    with open("run_start_time.txt", 'r') as f:
        start_time = f.read()

    for i in range(10):
        print(get_fitness_scores(".", np.loadtxt(f"generation_{i:02}/onset_times.txt"), start_time))