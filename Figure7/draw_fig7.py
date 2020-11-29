import sys,os
import numpy as np
sys.path.append('../')
from util import *
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
import seaborn as sns
sns.set_style("whitegrid")

## Broken Axis Configuration

# broken = [4, 6, 7]
broken = []
axis_broken_range = [() for i in range(8)]
axis_broken_range[3] = ((0.0, 0.002), (0.010, 0.020))
axis_broken_range[4] = ((0.0, 0.05), (0.80, 0.825))
axis_broken_range[6] = ((0.0, 0.15), (1.30, 1.4))
axis_broken_range[7] = ((0.0, 0.30), (1.875, 1.925))
if_broken = [i in broken for i in range(8)]

## Broken Axis Configuration End

localc = [(0.1 * x,0.1 * x,0.1 * x) for x in range(3,11,3)]

allnum = 2

features = ["DGL" , "PyG", "Ours", "Neighbor Grouping + Group Reorder","Neighbor Grouping + Group Reorder + thres 0.2"  ]
devs = [0 for i in range(8)]

fig = plt.figure(figsize=(12,4))
grid_shape = (2, 4)
grid = GridSpec(grid_shape[0], grid_shape[1])


featurelen = 32
estimate = []
yl = ["Time comparison (Linear)" , "DRAM Loads / Feature Mat Size"]
li = [[] for i in range(9)]
frameworks = ['dgl', 'pyg', 'our']
apps = ['gcn', 'gat', 'sage']
for i in range(9):
    if i == 7:
        li[7] = [1e-16 for i in range(8)]
        continue
    filename = "results/{}_{}_results.log".format(frameworks[i % 3], apps[i // 3])
    with open(filename, 'r') as f:
        res = f.readlines()
        for line in res:
            if "memory" in line or len(line) <= 1:
                thenum = 0
            else:
                thenum = (float)(line)
            if thenum < 1e-16:
                thenum = 1e-16
            li[i].append(thenum)


    
for iter,dev in enumerate(devs):
    #li = [[1,3,5], [2,4,5], [5,6,3]]
    #li = [[0.60299399277889, 0.693973297160197, 1.01288588881126], [0.335410774936195, 0.688314976756559, 0.488022780654868], [0.231411166801578, 0.619820118963201, 0.445978058540035]]
    #li = [[41.91,48.71,9.20],[45.96,43.62,9.46],[50.66,44.34,24.17],[79.18,58.67,53.70]]
    labels = ["arxiv", "collab", "citation", "ddi", "protein", "ppa", "reddit", "products"]
    
    # li[0] = [0.006031181,0.00850738,0.11516349,0.001130584,0.037959308,0.07427329,0.104694134,0.24603187]
    # li[1] = [0.0102569070556641,0.019805930246582,0.264695371246338,0.0123436778740234,0,0.309151680351257,0,0]
    # li[2] = [0.00345248,0.00518073,0.0688793,0.000912457,0.035485765,0.0306346,0.05808913,0.09255291]#[0.000290049,0.000525956,0.00696105,0.000171813,0.0053937,0.00488234,0.0103448,0.0153457]

    
    

    # li[3] = [0.017920713,0.028667512,0.44812862,0.018572839,0.806752842,0.46920901,1.325859222,1.90602484]
    # li[4] = [0.0110994810556641,0.021221369246582,0.309503026246338,0.0133463228740234,0,0.328839543351257,0,0]
    # li[5] = [0.004514228,0.006881258,0.08900158,0.000974787,0.034985702,0.03588354,0.055538462,0.11330854] # [0.002201428,0.003437518,0.05008908,0.00087438,0.033163437,0.02770754,0.052133032,0.08473463]# [0.000293836,0.000477013,0.00786744,0.000141591,0.00570035,0.0044223,0.00870771,0.0137715]
    # # for item in li[3]:
    # #     print(item)
    # # exit()
    
    

    # li[6] = [0.0145864486694336,0.0201017856597901,0,0.000646114349365235,0.0113966464996338,0.047050952911377,0.0193128585815429, 0]
    # li[7] = [1e-16 for i in range(8)]
    # li[8] = [0.0104168,0.0142771,0.178892,0.000321481,0.00857823,0.035739,0.0142037, 0]
    # #li[8] = [0.0104168,0.0142771,0.178892,0.000321481,0.00857823,0.035739,0.0142037, 0.1]
    
    whichdraw = iter

    yes = [[],[],[]]
    for i in range(3):
        for j in range(3):
            yes[i].append(li[i + j*3][whichdraw])
    x = np.arange(3)  # the label locations

    # if len(devs) == 1:
    #     ax = axs
    # else:
    #     ax = axs[iter // 4][iter%4]
    #     # print(type(ax))
    
    print('iter = ', iter)
    grd = grid[iter // grid_shape[1], iter % grid_shape[1]]
    xticklabel = ["", "", ""]
    if if_broken[iter]:
        ax = brokenaxes(ylims=axis_broken_range[iter], subplot_spec=grd)
        if(iter // grid_shape[1] == 1):
            xticklabel = ["","GCN","GAT","SAGE"]
    else:
        ax = fig.add_subplot(grd)
        if(iter // grid_shape[1] == 1):
            xticklabel = ["GCN","GAT","SAGE"]
    
    plt.title(labels[iter])
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabel)
    #if iter  == 0:
    #    ax.set_ylabel('Time(s)')
    #    ax.xaxis.set_label_coords(3.05, -0.025)
    width = 0.25 * 3 / len(yes)
    for it in range(len(yes)):
        ax.bar(x + it*width -  ((2 - (5-len(yes)) * 0.5)) * width, yes[it], width, color=localc[it], hatch=hatchs[it], edgecolor='k', label=features[it])
        #ax.bar(x + it*width -  ((2 - (5-len(yes)) * 0.5)) * width, yes[it], width, color=localc[it], edgecolor='k')

    # plt.legend(['cudnn','baseimpl', 'opt'], loc='upper left')
    # plt.ylim([0, max(cuda_opt + cuda) * 1.5])
    # if iter == 1:
    #     ax.legend( features + ["estimate using L2 cache size"] , loc='upper right',)
    # else:
    #     ax.legend( features , loc='upper right', ncol=3)
    # Setting the x-axis and y-axis limits

    ylimset = []

    ttmp = []
    for item in yes:
        ttmp += item
    mmax = max(ttmp)
    ttmp.remove(mmax)
    mmax2 = max(ttmp)
    thelimit = 100
    haha = []
    for item in ttmp:
        if item > 1e-10:
            haha.append(item)
    mmmin = min(haha) / 1.5
    if mmax / mmax2 > thelimit:
        ax.set_yscale("log")
        # ax.set_ylim([mmmin,  mmax * 1.5])
        ylimset = [mmmin,  mmax * 1.5]
    else:
        # ax.set_ylim([0,  mmax * 1.2])
        ylimset = [0,  mmax * 1.2]
    # ax.set_ylabel(yl[iter])

    if not if_broken[iter]:
        ax.set_ylim(ylimset)

    for i in range(3):
        li1 = []
        li2 = []
        for j in range(3):
            if yes[i][j] == 0:
                li1.append(j)
                li2.append(0)
        li1 = np.array(li1)
        for a,b in zip(li1 + i*width -  ((2 - (5-3) * 0.5)) * width,  li2 ):
            print(a,b)
            # if iter // 4 == 1:
                # b += 0.05
            if mmax / mmax2 > thelimit:
                b =mmmin 
            ax.text(a + 0.04, b, "OUT OF MEMORY", ha='center', va= 'bottom',rotation=90, fontsize=8)

    for i in range(3):
        li1 = []
        li2 = []
        for j in range(3):
            if yes[i][j] == 1e-16:
                li1.append(j)
                li2.append(0)
        li1 = np.array(li1)
        for a,b in zip(li1 + i*width -  ((2 - (5-3) * 0.5)) * width,  li2 ):
            print(a,b)
            # if iter // 4 == 1:
                # b += 0.05
            if mmax / mmax2 > thelimit:
                b = mmmin
            ax.text(a + 0.04, b, "OUT OF SUPPORT", ha='center', va= 'bottom',rotation=90, fontsize=8)
    if iter == 3:
        #ax.legend(features,loc='upper right',  bbox_to_anchor=(1.13, 1.2))
        ax.legend(loc='upper right',  bbox_to_anchor=(1.13, 1.2))

#cy =(0, max(tmp) * 1.3)
#plt.setp(axs, ylim=cy)

#plt.ylabel("Time(s)")
# plt.xlim(min(pos)-width, max(pos)+width*2.5)
# plt.ylim([0, max(cuda_opt + cuda) * 1.5])
# plt.legend(['base impl','sparse-sparse fuse on base impl','resource reallocation impl','sparse-sparse fuse on resource reallocation impl', 'e'], loc='upper left',fontsize=10)
# plt.grid()
fig.text(0.07, 0.5, 'Time (s)', va='center', rotation='vertical')
#fig.text(0.5, 0.03, 'Datasets', va='center', )

plt.show()
#plt.savefig(os.path.basename(__file__).split(".")[0] + ".pdf")
