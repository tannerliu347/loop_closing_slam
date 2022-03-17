# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

import seaborn as sns

# %%
data_path = "../data/"
result_path = data_path + "result.txt"
groud_truth_path = data_path + "kitti05GroundTruth.mat"

# %%
result = np.genfromtxt(result_path)
ground_truth = scipy.io.loadmat(groud_truth_path)
ground_truth = np.array(ground_truth["truth"])
for i in range(result.shape[0]):
    for j in range(result.shape[1]):
        if (result[i,j] == 1):
            result[j,i] = 1
result = np.tril(result, 0)
print(result[result==1].shape)
print(ground_truth[ground_truth==1].shape)
# %%
ax1 = plt.subplot(1, 2, 1)
ax2 = plt.subplot(1, 2, 2)
default_heatmap_kwargs = dict(
        xticklabels=False,
        yticklabels=False,
        square=True,
        cbar=False,)
sns.heatmap(ground_truth,
        ax=ax1,
        vmax=0.2,
        **default_heatmap_kwargs)
sns.heatmap(result,
        ax=ax2,
        vmax=0.2,
        **default_heatmap_kwargs)
# ax1.set_xlabel("current_frame")
# ax2.set_ylabel("matched_frame")
plt.show()

