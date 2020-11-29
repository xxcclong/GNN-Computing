#include "util.h"
#include "data.h"
#include "spmm.h"
#include "sample.h"
#include "aggr_gcn.h"
#include "aggr_gat.h"


enum PTRS
{
   x,
   y,
   y2,
   att,
   out_att,
   val
};

int main(int argc, char ** argv)
{
    const int times = 10;
    argParse(argc, argv);
    assert(GPUNUM == 1);
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);

    int* tmp1 = NULL;
    int* tmp2 = NULL;
    load_graph(inputgraph, n, m, tmp1, tmp2);
    gptrs = new int*[1];
    gidxs = new int*[1];
    checkCudaErrors(cudaMalloc2((void**)gptrs, (n + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)gidxs, m * sizeof(int)));
    checkCudaErrors(cudaMemcpy(gptrs[0], tmp1, sizeof(int) * (n + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(gidxs[0], tmp2, sizeof(int) * m, cudaMemcpyHostToDevice));

    vector<float*> ptr;
    vector<double> p_time(100, 10);
    vector<int> sizes = 
    {
        n * feature_len,
        n * feature_len,
        n * feature_len,
        n * 2,
        n * 2,
        m
    };

    for(auto item : sizes)
    {
        float* tmp = NULL;
        checkCudaErrors(cudaMalloc2((void**)&tmp, sizeof(float) * item));
        curandGenerateNormal(curand, tmp, item, 0.f, 1.00); 
        ptr.push_back(tmp);
    }

    int NEIGHBOR_NUM = 16;
    if(NEINUM != -1) NEIGHBOR_NUM = NEINUM;
    int BLOCK_SIZE = 128;

    dbg(NEIGHBOR_NUM);
    dbg(BLOCK_SIZE);

    auto g = fullGraph(gptrs[0], gidxs[0]);
    Aggregator_GCN * atgcn = new Aggregator_GCN(g, feature_len, feature_len, ptr[val]);
    Aggregator_GAT * atgat = new Aggregator_GAT(g, feature_len, feature_len);
    int tmparr[] = {NEIGHBOR_NUM};
    atgcn->schedule(neighbor_grouping, tmparr);
    atgat->schedule(neighbor_grouping, tmparr);

    // warm up
    for(int i = 0; i < times; ++i)
    {
        atgcn->run(ptr[x], ptr[y], BLOCK_SIZE, 0);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    // base implementation
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t_base0);
    for(int i = 0; i < times; ++i)
    {
        atgat->run_u_add_v(ptr[att], ptr[val], BLOCK_SIZE);
        atgat->run_add_to_center(ptr[val], ptr[out_att], BLOCK_SIZE);
        atgat->run_div_each(ptr[out_att], ptr[val], BLOCK_SIZE);
        atgcn->updateval(ptr[val]);
        atgcn->run(ptr[x], ptr[y], BLOCK_SIZE, 1);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    timestamp(t_base1);
    dbg(getDuration(t_base0, t_base1) / times);

    // using adapter to fuse
    timestamp(t_adapter0);
    for(int i = 0; i < times; ++i)
    {
        atgat->run_att(ptr[att], ptr[val], BLOCK_SIZE);
        atgcn->updateval(ptr[val]);
        atgcn->run(ptr[x], ptr[y2], BLOCK_SIZE, 1);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    timestamp(t_adapter1);
    dbg(getDuration(t_adapter0, t_adapter1) / times);

    timestamp(t_linear0);
    for(int i = 0; i < times; ++i)
    {
        atgat->run(ptr[x], ptr[att], ptr[y2], BLOCK_SIZE, 1);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    timestamp(t_linear1);
    dbg(getDuration(t_linear0, t_linear1) / times);
}
