import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TIME = {}
TIME['ct'] = {}
TIME['mr'] = {}
TIME['mr']['mesh'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Image_based2/test_ensemble_post/mr/time_results.csv'
TIME['ct']['mesh'] = '/Users/fanweikong/Documents/Modeling/SurfaceModeling/Image_based2/test_ensemble_post/time_results.csv'
TIME['mr']['seg'] = '/Users/fanweikong/Downloads/time_results_ct.csv'
TIME['ct']['seg'] = '/Users/fanweikong/Downloads/time_results_ct.csv'

SIZE = 20
OUT = {}
OUT['mr'] = [32]
OUT['ct'] = []
def plot_hist(data, labels, ax, modality):
    sns.set_style("white")
    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
    plt.figure(figsize=(10,7), dpi= 80)
    for i, col in enumerate(data):
        sns.distplot(col, label=labels[i], ax=ax, **kwargs)
        print("Max, median, min for %s: %1.2f, %1.2f, %1.2f" % (labels[i], np.max(col), np.median(col), np.min(col)))
        print("Avg pcercentage of total time for %s: %1.2f" % (labels[i], np.mean(col/np.sum(np.array(data), axis=0))*100)) 
    
    total = np.sum(np.array(data), axis=0)
    print("Max, median, min for total time: %1.2f, %1.2f, %1.2f" % (np.max(total), np.median(total), np.min(total)))
    ax.set_title(modality.upper(), size=SIZE+5)
    ax.set_xlim([-5,160])
    ax.set_ylim([0,0.3])
    ax.set_xlabel('Time (s)', fontsize=SIZE)
    ax.tick_params(labelsize=SIZE)

if __name__ == '__main__':

    modality = ['ct', 'mr']

    label = ['Segmentation', 'Post-Process', 'Geometry', 'Meshing']
    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    
    for i, m in enumerate(modality):
        fn_dict = TIME[m]
        mesh_t = np.loadtxt(open(fn_dict['mesh'], 'rb'), delimiter=",")
        seg_t = np.loadtxt(open(fn_dict['seg'], 'rb'), delimiter=",")
        if m=="ct":
            seg_t = seg_t[:40]
        else:
            seg_t = seg_t[40:80]

        for o in OUT[m]:
            seg_t = np.delete(seg_t, o)
        data = [seg_t] + [mesh_t[:,i] for i in range(mesh_t.shape[1])]
        plot_hist(data, label, axes[i], m)
    
    #handles, labels = axes[0].get_legend_handles_labels()    
    #fig.legend(handles, labels, loc = (0.5, 0.5), fontsize=SIZE)
    axes[-1].legend(loc='upper right', fontsize=SIZE)
    plt.show()
        
