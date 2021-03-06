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
    model_type = args.model
    print("model type={}".format(model_type))
    cudaid = args.gpu
    featurelen = args.feature_len
    torch.cuda.set_device(cudaid)
    device = torch.device("cuda:{}".format(cudaid))
    print("dset=" + str(args.syn_name))
    dset = args.syn_name
    #tdev = torch.cuda.get_device_name()
    #print(type(tdev))
    # load and preprocess dataset
    # data = load_data(args)
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
    # g = dgl.graph((src_list, dst_list)).to(device)
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

    weight0 = torch.randn([512, 128]).cuda()
    weight1 = torch.randn([128, 64]).cuda()
    weight2 = torch.randn([64, 32]).cuda()
    h = torch.randn([num_v, 512]).cuda()

    def gcn_layer_dgl_no_graph(feat, weight, in_feat_len, out_feat_len):
        feat2 = torch.mm(feat, weight)
        output = F.relu(feat2)
        torch.cuda.synchronize()
        return output
    def gat_layer_dgl_no_graph(feat, weight, attn_l, attn_r, in_feat_len, out_feat_len):
        feat2 = torch.mm(feat, weight)
        att_l = torch.mm(feat2, attn_l)
        att_r = torch.mm(feat2, attn_r)
        torch.cuda.synchronize()
        return feat2 

    def gcn_layer_dgl(feat, weight, in_feat_len, out_feat_len):
        feat2 = torch.mm(feat, weight)
        g.ndata['ft'] = feat2
        g.update_all(fn.copy_src(src='ft', out='m'),
                       fn.sum(msg='m', out='ft'))
        output = F.relu(g.dstdata['ft'])
        torch.cuda.synchronize()
        return output
    def gat_layer_dgl(feat, weight, attn_l, attn_r, in_feat_len, out_feat_len):
        feat2 = torch.mm(feat, weight)
        att_l = torch.mm(feat2, attn_l)
        att_r = torch.mm(feat2, attn_r)
        g.srcdata.update({'ft': feat2, 'el': att_l})
        g.dstdata.update({'er': att_r})
        g.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = torch.exp(F.leaky_relu(g.edata.pop('e'), 0.1))

        cont = utils.to_dgl_context(e.device)
        gidx = g._graph.get_immutable_gidx(cont)
        e_sum = backend.copy_reduce("sum", gidx, TargetCode.EDGE, e, num_v)
        att = backend.binary_reduce('none', 'div', gidx, TargetCode.EDGE, TargetCode.DST, e, e_sum, n_edges)
        g.edata['a'] = att
        g.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'))
        output = g.dstdata['ft']
        torch.cuda.synchronize()
        return output
    
    def sagelstm_layer_dgl(feat, lstm):
        def _lstm_reducer(nodes):
            m = nodes.mailbox['m'] # (B, L, D)
            batch_size = m.shape[0]
            h = (m.new_zeros((1, batch_size, 32)), # 32 is the hidden feature size
                m.new_zeros((1, batch_size, 32))) # 32 is the hidden feature size
            tin0 = time.time()
            _, (rst, _) = lstm(m, h)
            print("in lstm {}".format(time.time() - tin0))
            return {'neigh': rst.squeeze(0)}
        g.srcdata['h'] = feat
        tt0 = time.time()
        g.update_all(fn.copy_src('h', 'm'), _lstm_reducer)
        torch.cuda.synchronize()
        print("lstm {}", time.time() - tt0)
        return g.dstdata['neigh']

    runtimes = 10
    if model_type == "GCN":
        def GCN_forward():
            feat1 = gcn_layer_dgl(h, weight0, 512, 128)
            feat2 = gcn_layer_dgl(feat1, weight1, 128, 64)
            feat3 = gcn_layer_dgl(feat2, weight2, 64, 32)

        # warmup
        for it in range(runtimes):
            GCN_forward()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GCN_forward()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    elif model_type == "GAT":
        weight_l0 = torch.randn([128, 1]).cuda()
        weight_l1 = torch.randn([64, 1]).cuda()
        weight_l2 = torch.randn([32, 1]).cuda()

        weight_r0 = torch.randn([128, 1]).cuda()
        weight_r1 = torch.randn([64, 1]).cuda()
        weight_r2 = torch.randn([32, 1]).cuda()
        def GAT_forward():
            feat1 = gat_layer_dgl(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2 = gat_layer_dgl(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3 = gat_layer_dgl(feat2, weight2, weight_l2, weight_r2, 64, 32)

        # warmup
        for it in range(runtimes):
            GAT_forward()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GAT_forward()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    elif model_type == "sagelstm":
        lstm = nn.LSTM(32, 32, batch_first=True).cuda()
        sage_input = torch.randn([num_v, 32]).cuda()
        for it in range(runtimes):
            sagelstm_layer_dgl(sage_input, lstm)
        time0 = time.time()
        for it in range(runtimes):
            sagelstm_layer_dgl(sage_input, lstm)
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))


    elif model_type == "GAT_nograph":
        weight_l0 = torch.randn([128, 1]).cuda()
        weight_l1 = torch.randn([64, 1]).cuda()
        weight_l2 = torch.randn([32, 1]).cuda()

        weight_r0 = torch.randn([128, 1]).cuda()
        weight_r1 = torch.randn([64, 1]).cuda()
        weight_r2 = torch.randn([32, 1]).cuda()
        def GAT_forward_nograph():
            feat1 = gat_layer_dgl_no_graph(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2 = gat_layer_dgl_no_graph(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3 = gat_layer_dgl_no_graph(feat2, weight2, weight_l2, weight_r2, 64, 32)

        # warmup
        for it in range(runtimes):
            GAT_forward_nograph()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GAT_forward_nograph()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))
    elif model_type == "GCN_nograph":
        def GCN_forward_nograph():
            feat1 = gcn_layer_dgl_no_graph(h, weight0, 512, 128)
            feat2 = gcn_layer_dgl_no_graph(feat1, weight1, 128, 64)
            feat3 = gcn_layer_dgl_no_graph(feat2, weight2, 64, 32)

        # warmup
        for it in range(runtimes):
            GCN_forward_nograph()
        # run
        time0 = time.time()
        for it in range(runtimes):
            GCN_forward_nograph()
        print("DGL figure7 time {} {}".format(model_type, (time.time() - time0) / runtimes))




if __name__ == '__main__':
    print(dgl.__version__)
    print(dgl.__file__)
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--n-epochs", type=int, default=200,
            help="number of training epochs")
    parser.add_argument("--feature-len", type=int, default=32,
            help="number of hidden gcn units")
    parser.add_argument("--syn-name", type=str,
            help="")
    parser.add_argument("--model", type=str,
            help="", required=True)
    # parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
