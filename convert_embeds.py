import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA




def avg_embeds(embeds):
    avg_embed = np.mean(embeds, 0)
    print(np.size(avg_embed))

    return avg_embed

def do_pca(embeds):
    pca = PCA(n_components=np.size(embeds, 0))
    pcs = pca.fit_transform(embeds)
    print(pcs.shape)



if __name__ == "__main__":

    embeds = np.loadtxt("embeds_all.txt", delimiter=',')
    print(embeds.shape)

    avg_embed = avg_embeds(embeds)
    np.savetxt("avg_embeds.txt", avg_embed, delimiter=',')
    # do_pca(embeds)

    # # Create a histogram
    # plt.hist(avg_embed, bins=20, color='blue', edgecolor='black')

    # # Add labels and a title
    # plt.xlabel('Value')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Data')

    # # Show the plot
    # plt.show()





