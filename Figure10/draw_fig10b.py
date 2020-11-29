
import sys,os
import numpy as np
sys.path.append('../')
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

localc = [(0.1 * x,0.1 * x,0.1 * x) for x in range(4,11,5)]
features = ["Base" , "Base+Adapter+Linear", "NS", "NS+Adapter"  ]
fig, ax = plt.subplots(1, 1,figsize=(6,2))

featurelen = 32
labels = ["arxiv", "collab", "citation", "ddi", "protein", "ppa", "reddit", "products"]


filenames = ["results/fig10b_base.log", "results/fig10b_base_adapter_linear.log"]
li = [[] for i in range(len(filenames))]
for fit, filename in enumerate(filenames):
    with open(filename, 'r') as f:
        res = f.readlines()
        for line in res:
            li[fit].append((float)(line))


pos = list(range(len(li[0]))) 
x = np.arange(len(labels))  # the label locations

for i in range(len(labels)):
    mmax = max(li[0][i],li[1][i])
    for j in range(2):
        li[j][i] /= mmax

width = 0.25 * 3 / len(li)
for it in range(len(li)):
    ax.bar(x + it*width -  ((2 - (5-len(li)) * 0.5)) * width, li[it], width, color=localc[it], hatch=hatchs[it], edgecolor='k')

ax.title.set_text("")
ax.set_xticks(x)
ax.set_xticklabels(labels, )
ax.legend( features , ncol=2,)
ttmp = []
for i in range(len(li)):
    ttmp += li[i]

ax.set_ylim([0, max(ttmp) * 1.4])
ax.set_ylabel("Running time(Normalized)")


plt.savefig(os.path.basename(__file__).split(".")[0][5:] + ".pdf")