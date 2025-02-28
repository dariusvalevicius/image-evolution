from datetime import datetime, timedelta
import numpy as np
import pandas as pd



def get_fitness_scores(onset_times, start_time, ratings=None):
    '''
        This is a modular function that is called at the end of every generation.

        GENERATOR.PY will pass it the stimulus onset times, as well as the subjective ratings if applicable.

        onset_times is the onset times in seconds relative to the initiation of the program (that is, when the participant hit the spacebar after reading the instruction text).

        start_time is passed as a datetime object. Use this to align the stimulus onset times to your recording device, if applicable.
        As such, datetime is a reccomended package for this function.
    '''

    fitness = np.zeros(onset_times.shape[0])
    start_time_dt = datetime.strptime(start_time, "%m_%d_%Y_%H_%M_%S_%f")

    ######################################################################################################
    # YOUR CODE HERE #
    ######################################################################################################
    ## In this example, the fitness function is simply the subjective ratings submitted by the participant.
    fitness = ratings






    ######################################################################################################
    # END #
    ######################################################################################################

    return fitness