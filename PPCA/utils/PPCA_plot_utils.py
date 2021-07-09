import os
import matplotlib.pyplot as plt
import numpy as np
def plot_participation_ratio(pca_object=None, ax = None, plot_folder=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(np.arange(1,len(pca_object.explained_variance_ratio_)+1),
             np.cumsum(pca_object.explained_variance_ratio_))
    ax.grid()
    ax.set_ylabel('frac. explained var.')
    ax.set_xlabel('# components')
    ax.set_xticks(np.arange(1,len(pca_object.explained_variance_ratio_)+1))
    ax.set_title('PCA participation ratio')
    return ax
    # if plot_folder is not None:
    #     plt.savefig(os.path.join(plot_folder, 'participation_ratio.png'))

def compare_post_cov_norm(predictions, missing_obs_bool, counts, ax):
    if ax is None:
        fig, ax = plt.subplots()
    strs = ["miss %i/%i" % (i, counts[-1]) for i in counts]
    for i in range(len(counts)):
        post_missing = predictions["cov"][:,:,missing_obs_bool[i,:]]
        norms_missing = np.linalg.norm(post_missing, axis=(0,1))
        ax.scatter(np.ones(len(norms_missing))*counts[i], norms_missing)
    #missing_obs_bool = np.sum(np.isnan(arr_squeezed), axis=0)>0
    ax.set_xticks(counts)
    ax.set_xticklabels(strs)
    ax.set_ylabel(r"$||Cov(z|x)||$")
    ax.set_title('Post. Cov. as f(missing views)')
    return ax

def set_legend_elements(markers, colors, labels):
    from matplotlib.lines import Line2D
    legend_elements = []
    for i in range(len(markers)):
        legend_elements.append(Line2D([0], [0], marker=markers[i], color=colors[i], label=labels[i],
                markersize=10, linestyle = None, linewidth=0))
    return legend_elements
