import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

#plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
#import matplotlib
#font = {'family' : 'normal',
#        'weight' : 'bold',
#        'size'   : 22}
#
#matplotlib.rc('font', **font)
SIZE = 15
def fix_labels(dir_n):
    import SimpleITK as sitk
    ids = [0, 205, 420, 500, 550, 600, 820, 850] 
    
    
    fns = sorted(glob.glob(os.path.join(dir_n, '*.nii.gz')))
    
    for n in fns:
        im = sitk.ReadImage(n)
        pyarr = sitk.GetArrayFromImage(im)
        for index, label in enumerate(ids):
            pyarr[pyarr==index]= label
        im_new = sitk.GetImageFromArray(pyarr)
        im_new.SetOrigin(im.GetOrigin())
        im_new.SetDirection(im.GetDirection())
        im_new.SetSpacing(im.GetSpacing())
    
        sitk.WriteImage(im_new, n)
    
def fix_excels(dir_n, metric):
    modality = ["ct", "mr"]
    results = {}
    for m in modality:
        results[m.upper()] = []
        fns = sorted(glob.glob(os.path.join(dir_n, m+'*'+metric+'*.xls')))
        out = open(os.path.join(dir_n, m+'_'+metric+'.xls'), 'w+')
        for fn in fns:
            print(fn)
            with open(fn) as f:
                for line in f:
                    score = []
                    for s in line.split():
                        try:
                            score.append(float(s))
                        except: pass
                    results[m.upper()].append(score)
                    out.write(line)
        results[m.upper()] = np.array(results[m.upper()])
        out.close()
    return results

def fix_excels_surface(dir_n):
    modality = ["ct", "mr"]
    classes = ["LV", "Epi", "RV", "LA", "RA", "LO", "PA", "WHS"]
    #classes = ["WHS"]
    results = {}
    for m in modality:
        results[m.upper()] = []
        for c in classes:
            out = open(os.path.join(dir_n, m+ '_surfaces_'+c + '.xls'), 'w+')
            score = []
            for i in range(1, 41):
                fn = os.path.join(dir_n, m+'_surface'+str(i)+'_'+c+'.xls')
                print(fn)
                with open(fn) as f:
                    for line in f:
                        try:
                            float(line.split()[0])
                            try:
                                score.append(float(line.split()[1]))
                            except Exception as e: pass
                            out.write(line)
                        except Exception as e:
                            #print(e)
                            pass
            results[m.upper()].append(score)
            out.close()
        results[m.upper()] = np.array(results[m.upper()]).transpose()


    return results
    
#def plot_histogram(data, labels):
#    # data a list containing vectors of scores
#    # label, a list of data labels
#    import seaborn as sns
#    sns.set_style("white")
#    kwargs = dict(hist_kws={'alpha':.6}, kde_kws={'linewidth':2})
#    plt.figure(figsize=(10,7), dpi= 80)
#    for i, col in enumerate(data):
#        sns.distplot(col, label=labels[i], **kwargs)
#    plt.legend()
#    plt.show()

def plot_boxplot(data, axes, classes, ids, y_label):
    import pandas as pd
    import seaborn as sns
    
    dfs = []
    keys = list(data.keys())
    for k in keys:
        try:
            dfs.append(pd.DataFrame([data[k][:, i] for i in ids], 
                [classes[i] for i in ids]).T)
        except:
            dfs.append(pd.DataFrame([data[k][i][:] for i in ids], 
                [classes[i] for i in ids]).T)
        
    sns.set(style="whitegrid")
    for i, ax in enumerate(axes):
        sns.boxplot(data=dfs[i], linewidth=1., ax=ax)
        if i==0:
            ax.set_title(y_label, size=SIZE+5)
        ax.set_ylim([0.,np.ceil(np.nanmax(dfs[i].values))])
        ax.yaxis.set_major_formatter(FormatStrFormatter('%0.1f'))
        ax.tick_params(labelsize=SIZE)

def plot_table(dice, jaccard, surfd, classes, ids):
    import pandas as pd
    paper = {}
    paper_std = {}
    paper['CT'] = [0.908,0.086, 0.823,0.037, 1.117, 0.250] 
    paper['MR'] = [0.874, 0.039, 0.778, 0.06, 1.631, 0.58]

    keys = list(dice.keys())
    DFS = []
    for k in keys:
        dice_df = pd.DataFrame([dice[k][:,i] for i in ids], 
                [classes[i] for i in ids]).T
        dice_mean = dice_df.mean()
        dice_std = dice_df.std()
        jc_df = pd.DataFrame([jaccard[k][:,i] for i in ids], 
                [classes[i] for i in ids]).T
        jc_mean = jc_df.mean()
        jc_std = jc_df.std()
        surf_df = pd.DataFrame([surfd[k][i][:] for i in ids], 
                [classes[i] for i in ids]).T
        surf_mean = surf_df.mean()
        surf_std = surf_df.std()
        
        DF = pd.concat([dice_mean, dice_std, jc_mean, jc_std, surf_mean, surf_std], axis=1).T
        print(DF)
        DF['WH paper'] = pd.Series(paper[k], index=DF.index)
        DF = DF.T
        DF = DF.round(3)
        DF.columns = ['Dice','Dice_std', 'Jaccard', 'Jaccard_std','SD (mm)', 'SD (mm)_std']
        print(DF.columns.str.split('_').str[0])
        DF = DF.groupby(DF.columns.str.split('_').str[0], axis=1).apply(lambda x: x.astype(str).apply('$\pm$'.join, 1))
        DFS.append(DF.T)
        print(DF)
    table_df = pd.concat([d for d in DFS], keys=keys)
    print(table_df)
    print(table_df.to_latex(index = True, multirow = True,multicolumn=True, escape=False))
if __name__ =='__main__':
    #fix_labels()
    dir_n = '/Users/fanweikong/Downloads/test_ensemble_post-1'
    jaccard = fix_excels(dir_n, 'jaccard')
    dice = fix_excels(dir_n, 'dice')
    surface = fix_excels_surface(dir_n)
    
    # plotting dice of LV (0), LA(3), Aorta(5), Whole heart(7)
    classes = ["LV", "Epi", "RV", "LA", "RA", "Aorta", "PA", "WH"] 
    ids = [0,3,5,7]
    
    #fig, axes = plt.subplots(2, 3, figsize=(12,12))
    ##fig.tight_layout()
    #fig.subplots_adjust(hspace=.2)
    #plot_boxplot(dice,axes[:,0], classes, ids, 'Dice scores')
    #plot_boxplot(jaccard,axes[:,1], classes, ids, 'Jaccard scores')
    #plot_boxplot(surface,axes[:,2], classes, ids, 'Surface errors (mm)')
    #axes[0,0].set_ylabel('CT', fontsize=SIZE+5, rotation=0)
    #axes[0,0].yaxis.set_label_coords(-0.3,0.5)
    #axes[1,0].set_ylabel('MR', fontsize=SIZE+5, rotation=0)
    #axes[1,0].yaxis.set_label_coords(-0.3,0.5)
    ##plt.show()
    #plt.savefig(os.path.join(dir_n, 'results.png'))
    
    plot_table(dice, jaccard, surface, classes, ids)


