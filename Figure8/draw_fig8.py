import os
import sys
import numpy as np
sys.path.append('../')
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
localc = ['w', 'grey']


features = ["Base Balanced", "Base Actual", "NG Balanced", "NG Actual"]
fig, ax = plt.subplots(1, 1,figsize=(6,3))


labels = ["arxiv", "collab", "citation", "ddi", "protein", "ppa", "reddit", "products"]
dsetnum = len(labels)


filenames = ["results/fig8_base_actual.log", "results/fig8_base_balanced.log", "results/fig8_NG_actual.log", "results/fig8_NG_balanced.log"]
li = [[] for i in range(len(filenames))]
for fit, filename in enumerate(filenames):
    with open(filename, 'r') as f:
        res = f.readlines()
        for line in res:
            li[fit].append((float)(line))
    

width = 0.25 * 3 / 4 * 2
x = np.arange(dsetnum)  # the label locations
drawlist = [i for i in range(4)]
mmax = [li[0][i] for i in range(dsetnum)]
for it in range((2)):
    tmp = [0 for kkkk in range(dsetnum)]
    for i in range(dsetnum):
        li[it * 2 + 1][i] /= mmax[i]
        li[it * 2][i] /= mmax[i]
        tmp[i] = li[it * 2][i] - li[it * 2 + 1][i]
    ax.bar(x + it*width -  ((2 - (5-len(li) // 2) * 0.5)) * width, li[it * 2 + 1] , width, color=localc[0], hatch=hatchs[it], edgecolor='k')
    ax.bar(x + it*width -  ((2 - (5-len(li) // 2) * 0.5)) * width, tmp, width, color=localc[1], bottom= li[it * 2 + 1], hatch=hatchs[it], edgecolor='k')

ax.title.set_text("")
ax.set_xticks(x)
ax.set_xticklabels(labels, )
ttmp = [features[i] for i in drawlist]
ax.legend( ttmp , loc="lower right", ncol = 1)
ax.set_ylabel("Time (Relative)")
ax.set_xlabel("Datasets")



plt.savefig(os.path.basename(__file__).split(".")[0][5:] + ".pdf")
