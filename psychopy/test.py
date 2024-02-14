import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def downsample_to_target_samples(arr, target_samples):

    arr = arr.values
    # Calculate the average pooling factor
    factor = arr.shape[0] // target_samples
    
    # Reshape the array into blocks
    reshaped_arr = arr[:target_samples * factor].reshape(-1, factor)
    
    # Take the average value within each block
    downsampled_arr = np.mean(reshaped_arr, axis=1)
    
    return downsampled_arr

# Example usage
# original_data = np.array([1, 2, 3, 4, 5, 6, 7, 6, 3, 4, 6, 7, 8, 9, 10, 9, 7, 6, 3, 5, 7, 3, 1, 4, 5,2, 4, 6, 8, 6])
original_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 6, 3, 4, 6, 7, 8, 9, 10, 9, 7, 6, 3, 5, 7, 3, 1, 4, 5,2, 4, 6, 8, 6])
print(original_data.shape)
plt.plot(original_data)
plt.show()
target_samples = 25
downsampled_data = downsample_to_target_samples(original_data, target_samples)
print(downsampled_data)
print(downsampled_data.shape)

plt.plot(downsampled_data)
plt.show()
