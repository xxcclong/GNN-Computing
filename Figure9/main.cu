#include "util.h"
#include "data.h"
#include "spmm.h"
#include "sample.h"
#include "aggr_gcn.h"

enum PTRS
{
   x,
   y,
   y2,
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
    int BLOCK_SIZE = 512;

    dbg(NEIGHBOR_NUM);
    dbg(BLOCK_SIZE);

    auto g = fullGraph(gptrs[0], gidxs[0]);
    Aggregator_GCN * atgcn = new Aggregator_GCN(g, feature_len, feature_len, ptr[val]);
    int tmparr[] = {NEIGHBOR_NUM};
    atgcn->schedule(neighbor_grouping, tmparr);

    for(int i = 0; i < times; ++i)
    {
        atgcn->run(ptr[x], ptr[y], BLOCK_SIZE, 0);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    for(int i = 0; i < times; ++i)
    {
        atgcn->run(ptr[x], ptr[y2], BLOCK_SIZE, 1);
        checkCudaErrors(cudaDeviceSynchronize());
    }
}
