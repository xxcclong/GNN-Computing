import torch, time, sys
from typing import Optional, Tuple
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_add
import pickle, os
import argparse
@torch.jit.script
def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


@torch.jit.script
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = src.size()
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = 0 
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        out = out.scatter_add_(dim, index, src)
        return out
    else:
        out = out.scatter_add_(dim, index, src)
        return out

def main(args):
    filename = args.dset 
    cudaid = args.gpu
    model_type = args.model
    torch.cuda.set_device(cudaid)
    basedir = "../data/"
    cachedir = "../data_pyg/"
    if not os.path.exists(cachedir):
        os.makedirs(cachedir)
    edgeindex = [[],[]]
    cachefile = cachedir + filename + ".pkl"
    configf = open(basedir + filename + ".config", "r")
    l = configf.readline()
    num_v = (int)(l.split(' ')[0])
    num_e = (int)(l.split(' ')[1].strip("\n"))
    configf.close()

    if os.path.isfile(cachefile):
        edgeindex = pickle.load(open(cachefile, 'rb'))
    else:
        graphf = open(basedir + filename + ".graph", "r")
        l = graphf.readline().strip("\n").split(' ')
        l2 = graphf.readline().strip("\n").split(' ')
        if "" in l:
            l.remove("")
        if "" in l2:
            l2.remove("")

        outd = [[] for i in range(num_v)]
        for i in range(num_v):
            begin = (int)(l[i])
            end = (int)(l[i + 1])
            for j in range(begin, end):
                outd[(int)(l2[j])].append(i)

        for it, item in enumerate(outd):
            edgeindex[0] += item
            edgeindex[1] += [it for i in range(len(item))]
        df = open(cachefile, "wb")
        pickle.dump(edgeindex, df)
        df.close()
    edgeindex = torch.tensor(edgeindex).cuda()
    torch.manual_seed(123)
    weight0 = torch.randn([512, 128]).cuda()
    weight1 = torch.randn([128, 64]).cuda()
    weight2 = torch.randn([64, 32]).cuda()
    h = torch.randn([num_v, 512]).cuda()

    def gcn_layer_pyg(feat, weight, in_feat_len, out_feat_len):
        freetime = 0
        feat2 = torch.mm(feat, weight)
        src = torch.index_select(feat2, 0, edgeindex[1])
        output = torch.zeros(feat2.shape, dtype=src.dtype, device=src.device)
        output = F.relu(scatter_sum(src, edgeindex[0], 0, output))
        torch.cuda.synchronize()

        t0 = time.time()
        del src
        del feat2
        if feat.shape[1] != 512:
            del feat
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freetime += time.time() - t0

        return output, freetime

    def gat_layer_pyg(feat, weight, attn_l, attn_r, in_feat_len, out_feat_len):
        freetime = 0
        feat2 = torch.mm(feat, weight)
        att_l = torch.mm(feat2, attn_l)
        att_r = torch.mm(feat2, attn_r)
        att_l_edge = torch.index_select(att_l, 0, edgeindex[1])
        att_r_edge = torch.index_select(att_r, 0, edgeindex[1])
        att = att_l_edge + att_r_edge
        att = att / (scatter_add(att, edgeindex[0], dim=0, dim_size=num_v)[edgeindex[0]] + 1e-16)
        src = torch.index_select(feat2, 0, edgeindex[1]) * (att.view(-1, 1))
        output = torch.zeros(feat2.shape, dtype=src.dtype, device=src.device)
        output = F.leaky_relu(scatter_sum(src, edgeindex[0], 0, output), 0.1)
        torch.cuda.synchronize()

        t0 = time.time()
        del att_l 
        del att_r 
        del att_l_edge
        del att_r_edge
        del att
        del src
        del feat2
        if feat.shape[1] != 512:
            del feat
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        freetime += time.time() - t0

        return output, freetime


    runtimes = 30
    if model_type == "GCN":

        freetime = 0
        # warmup
        #for it in range(runtimes):
        #    feat1 = gcn_layer_pyg(h, weight0, 512, 128)
        #    feat2 = gcn_layer_pyg(feat1, weight1, 128, 64)
        #    feat3 = gcn_layer_pyg(feat2, weight2, 64, 32)
        # run
        time0 = time.time()
        for it in range(runtimes):
            feat1, t1 = gcn_layer_pyg(h, weight0, 512, 128)
            # print("allocated", torch.cuda.memory_allocated("cuda:{}".format(cudaid)))
            feat2, t2 = gcn_layer_pyg(feat1, weight1, 128, 64)
            # print("allocated", torch.cuda.memory_allocated("cuda:{}".format(cudaid)))
            feat3, t3 = gcn_layer_pyg(feat2, weight2, 64, 32)
            # print("allocated", torch.cuda.memory_allocated("cuda:{}".format(cudaid)))
            freetime += t1
            freetime += t2
            freetime += t3

        print("PYG figure7 time {} {}".format(model_type, (time.time() - time0 - freetime) / runtimes))
    elif model_type == "GAT":
        weight_l0 = torch.randn([128, 1]).cuda()
        weight_l1 = torch.randn([64, 1]).cuda()
        weight_l2 = torch.randn([32, 1]).cuda()

        weight_r0 = torch.randn([128, 1]).cuda()
        weight_r1 = torch.randn([64, 1]).cuda()
        weight_r2 = torch.randn([32, 1]).cuda()

        freetime = 0

        # warmup
        for it in range(runtimes):
            feat1, t1 = gat_layer_pyg(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2, t2 = gat_layer_pyg(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3, t3 = gat_layer_pyg(feat2, weight2, weight_l2, weight_r2, 64, 32)
        # run
        time0 = time.time()
        for it in range(runtimes):
            feat1, t1 = gat_layer_pyg(h, weight0, weight_l0, weight_r0, 512, 128)
            feat2, t2 = gat_layer_pyg(feat1, weight1, weight_l1, weight_r1, 128, 64)
            feat3, t3 = gat_layer_pyg(feat2, weight2, weight_l2, weight_r2, 64, 32)
            freetime += t1
            freetime += t2
            freetime += t3

        print("PYG figure7 time {} {}".format(model_type, (time.time() - time0 - freetime) / runtimes))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    parser.add_argument("--gpu", type=int, default=0,
            help="gpu")
    parser.add_argument("--dset", type=str,
            help="")
    parser.add_argument("--model", type=str,
            help="", required=True)
    # parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)

    main(args)
