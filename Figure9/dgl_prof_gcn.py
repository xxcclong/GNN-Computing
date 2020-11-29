import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
import dgl.function as fn
from dgl.function import TargetCode
import dgl.backend as backend
import dgl.utils as utils 
import dgl

def main(args):
    cudaid = args.gpu
    featurelen = args.feature_len
    torch.cuda.set_device(cudaid)
    device = torch.device("cuda:{}".format(cudaid))
    print("dset=" + str(args.dset))
    dset = args.dset
    num_v = 0
    num_e = 0
    with open("../data/{}.config".format(dset), 'r') as f:
        l = f.readline().split(' ')
        num_v = (int)(l[0])
        num_e = (int)(l[1])

    src_list = []
    dst_list = []
    t_load_begin = time.time()
    with open("../data/{}.graph".format(dset), 'r') as f:
        ptr = f.readline().strip("\n").strip(" ").split(" ")
        idx = f.readline().strip("\n").strip(" ").split(" ")
        for item in range(num_v):
            which = (int)(item)
            selfloop = False
            for i in range((int)(ptr[which]), (int)(ptr[which + 1])):
                dst_list.append(which)
                src_list.append((int)(idx[i]))
                if which == (int)(idx[i]):
                    selfloop = True
            # if not selfloop:
            #     dst_list.append(which)
            #     src_list.append(which)
    # print("load data time {}".format((str)(time.time() - t_load_begin)))


    g = DGLGraph((src_list, dst_list)).to(device)
    cuda = True

    #g.remove_edges_from(nx.selfloop_edges(g))
    # add self loop
    #if args.self_loop:
    #    g.remove_edges_from(nx.selfloop_edges(g))
    #    g.add_edges_from(zip(g.nodes(), g.nodes()))

    # g = DGLGraph(g)
    n_edges = g.number_of_edges()
    num_v = g.number_of_nodes()
    torch.manual_seed(123)

    h = torch.randn([num_v, featurelen]).cuda()

    def gcn_last_layer_graph_op(feat):
        g.ndata['ft'] = feat
        g.update_all(fn.copy_src(src='ft', out='m'),
                       fn.sum(msg='m', out='ft'))


    runtimes = 1
    # run
    for it in range(runtimes):
        gcn_last_layer_graph_op(h)

if __name__ == '__main__':
    print(dgl.__version__)
    print(dgl.__file__)
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--feature-len", type=int, default=32,
            help="number of hidden gcn units")
    parser.add_argument("--dset", type=str,
            help="")
    # parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
