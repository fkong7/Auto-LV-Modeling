import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_xls(fn, num_class=8):
    
    scores = np.zeros((0, num_class))
    with open(fn, 'r') as f:
        for line in f:
            line = line.split()
            value = []
            for i in line:
                try: 
                    value.append(float(i))
                except:
                    pass
            scores = np.vstack((scores, np.array(value)))
    return scores

def box_plot(scores, attr):
    df = pd.DataFrame(scores, columns=attr)
    boxplot = df.boxplot()
    plt.ylim([0,1])
    plt.show()

if __name__ == '__main__':
    fns = ['/Users/fanweikong/Downloads/mr_dice_ensemble_aug.xls',
            '/Users/fanweikong/Downloads/mr_dice_ensemble.xls',
            '/Users/fanweikong/Downloads/ct_dice_ensemble_aug.xls',
            '/Users/fanweikong/Downloads/ct_dice_ensemble.xls'
            ]
    attr = ['LV', 'Myo', 'RV', 'LA', 'RA', 'AO', 'PA', 'WHS'] 
    for fn in fns:
        scores = parse_xls(fn)
        box_plot(scores, attr)
