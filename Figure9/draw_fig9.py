import sys,os
import numpy as np
sys.path.append('../')
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

features = ["Best prior" , "NG", "LAS", "NG+LAS","NG+RO2"  ]
fig, ax = plt.subplots(1, 1,figsize=(6,3))

filenames = ["results/fig9_best_prior.log", "results/fig9_NG.log", "results/fig9_LAS.log", "results/fig9_NG_LAS.log"]
li = [[] for i in range(len(filenames))]
for fit, filename in enumerate(filenames):
    with open(filename, 'r') as f:
        res = f.readlines()
        for line in res:
            li[fit].append((float)(line))

width = 0.25 * 3 / len(li)

pos = list(range(len(li[0]))) 
labels = ['arxiv', 'collab', 'citation', 'ddi', 'protein', 'ppa', 'reddit', 'products']
x = np.arange(len(labels))  # the label locations
for it in range(len(li)):
    ax.bar(x + it*width -  ((2 - (5-len(li)) * 0.5)) * width, li[it], width, color=c[it], hatch=hatchs[it], edgecolor='k')
ax.title.set_text("")
ax.set_xticks(x)
ax.set_xticklabels(labels, )
ax.legend( features , loc='upper left',bbox_to_anchor=(0, 1.05) )
ttmp = []
for item in li:
    ttmp += item

ax.set_ylim([0, max(ttmp) * 1.2])
ax.set_ylabel("L2 Cache Hit Rate (%)")
ax.set_xlabel("Datasets")

 
plt.savefig(os.path.basename(__file__).split(".")[0][5:] + ".pdf")
