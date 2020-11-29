import sys,os
import numpy as np
sys.path.append('../')
from util import *
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

features = ["Base", "+SpFetch", "+RedBypass"]
fig, ax = plt.subplots(1, 1,figsize=(6,2))

featurelen = 32
estimate = []
labels = ["arxiv", "collab", "citation", "ddi", "protein", "ppa", "reddit", "products"]


filenames = ["results/fig11_base.log", "results/fig11_SF.log", "results/fig11_SF_RE.log"]
li = [[] for i in range(len(filenames))]
for fit, filename in enumerate(filenames):
    with open(filename, 'r') as f:
        res = f.readlines()
        for line in res:
            li[fit].append((float)(line))


width = 0.25 * 3 / len(li)

pos = list(range(len(li[0]))) 
x = np.arange(len(labels))  # the label locations

for i in range(len(labels)):
    mmax = max(li[0][i],li[1][i],li[2][i])
    for j in range(3):
        li[j][i] /= mmax + 1e-16


to_draw = [0,1,2]
for it in range(len(li)):
    ax.bar(x + it*width -  ((2 - (5-len(li)) * 0.5)) * width, li[to_draw[it]], width, color=c[it], hatch=hatchs[it], edgecolor='k')

ax.title.set_text("")
ax.set_xticks(x)
ax.set_xticklabels(labels, )
ax.legend( features ,ncol=2,)
ttmp = []
for item in li:
    ttmp += item

ax.set_ylim([0, max(ttmp) * 1.7])
ax.set_ylabel("Running time(Normalized)")


plt.savefig(os.path.basename(__file__).split(".")[0][5:] + ".pdf")
