#include "util.h"
#include "data.h"
#include "spmm.h"
#include "sample.h"
#include "aggr_gcn.h"
#include "dense.h"


enum PTRS
{
   x,
   y,
   y2,
   val,
   weight,
   transformed,
   transformed2,
   tmp
};

int main(int argc, char ** argv)
{
    const int times = 10;
    argParse(argc, argv);
    const int out_feature_len = outfea;
    assert(out_feature_len > 0);
    assert(GPUNUM == 1);
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    cublasCreate(&cublasHs[0]);

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
        n * out_feature_len,
        n * out_feature_len,
        m,
        feature_len * out_feature_len,
        n * out_feature_len,
        n * out_feature_len,
        n * out_feature_len,
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
    int tmparr[] = {NEIGHBOR_NUM};
    atgcn->schedule(neighbor_grouping, tmparr);

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
        atgcn->run(ptr[x], ptr[y2], BLOCK_SIZE, 1);
        matmul_NN(ptr[y2], ptr[weight], ptr[transformed2], n, out_feature_len, feature_len, ptr[tmp]);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    timestamp(t_base1);
    dbg(getDuration(t_base0, t_base1) / times);

    timestamp(t_linear0);
    for(int i = 0; i < times; ++i)
    {
        atgcn->run_with_nn(ptr[x], ptr[y], ptr[weight], ptr[transformed], BLOCK_SIZE);
        checkCudaErrors(cudaDeviceSynchronize());
    }
    timestamp(t_linear1);
    dbg(getDuration(t_linear0, t_linear1) / times);
}
