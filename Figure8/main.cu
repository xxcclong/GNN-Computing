//#include "util.h"
//#include "data.h"
//#include "aggr_kernel_no_template.h"
//#include "spmm.h"
//#include "dense.h"
#include <queue>

#include "util.h"
#include "data.h"
#include "spmm.h"
// #include "gtime.h"
#include "sample.h"
#include "aggr_gcn.h"
// #include "aggr_gat.h"
// #include "aggr_nn.h"
// #include "aggr_sddmm.h"
// #include "partition.h"

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
    const int att_len = 20;
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
    int BLOCK_SIZE = 64;

    dbg(NEIGHBOR_NUM);
    dbg(BLOCK_SIZE);

    auto g = fullGraph(gptrs[0], gidxs[0]);
    Aggregator_GCN * atgcn = new Aggregator_GCN(g, feature_len, feature_len, ptr[val]);
    int tmparr[] = {NEIGHBOR_NUM};
    atgcn->schedule(neighbor_grouping, tmparr);

    int tmp_target_in_block = BLOCK_SIZE / feature_len;
    const dim3 fine_grid_expand_y((atgcn->num_target + tmp_target_in_block - 1) / tmp_target_in_block);
    const dim3 fine_block_expand_y(BLOCK_SIZE / feature_len * 32, feature_len / 32);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int numBlocksPerSm = -1;
    int numBlocksPerSm2 = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, aggr_gcn_target_clock, fine_block_expand_y.x * fine_block_expand_y.y, 0);

    const dim3 coarse_grid_expand_y((n + tmp_target_in_block - 1) / tmp_target_in_block);
    const dim3 coarse_block_expand_y(BLOCK_SIZE / feature_len * 32, feature_len / 32);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm2, aggr_gcn_clock, coarse_block_expand_y.x * coarse_block_expand_y.y, 0);
    dbg(numBlocksPerSm);
    dbg(numBlocksPerSm2);

    clocktype* thetimer;
    checkCudaErrors(cudaMalloc2((void**)&thetimer, sizeof(clocktype) * 3 * fine_grid_expand_y.x ));

    clocktype* thetimer2;
    checkCudaErrors(cudaMalloc2((void**)&thetimer2, sizeof(clocktype) * 3 * coarse_grid_expand_y.x));
    dbg(fine_grid_expand_y.x);
    
    clocktype c_begin = 0;
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaMemset(ptr[y], 0, feature_len * n * sizeof(float)));
    checkCudaErrors(cudaMemset(thetimer, 0, 3 * fine_grid_expand_y.x * sizeof(clocktype)));
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t0);
    // finegrain_shared_atomic_no_template_clock<<<  fine_grid_expand_y, fine_block_expand_y, NEIGHBOR_NUM * 2 * tmp_target_in_block * sizeof(int)  >>>
    //     (targetv_d, newptr, gidxs[0], gvals[0], ptr[x], ptr[y], target_v.size(), feature_len, NEIGHBOR_NUM, thetimer);
    atgcn->run_clock(ptr[x], ptr[y], thetimer, BLOCK_SIZE, 1);
    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t1);

    // coarsegrain_shuffle_nonatomic_no_template_clock <<< coarse_grid_expand_y, coarse_block_expand_y >>>
        // (gptrs[0], gidxs[0], gvals[0], ptr[x], ptr[y2], n, feature_len, thetimer2);
    atgcn->run_clock(ptr[x], ptr[y2], thetimer2, BLOCK_SIZE, 0);

    checkCudaErrors(cudaDeviceSynchronize());
    timestamp(t2);

    clocktype* cpu_timer = new clocktype[3 * fine_grid_expand_y.x];
    memset(cpu_timer, 0, 3 * fine_grid_expand_y.x * sizeof(clocktype));
    checkCudaErrors(cudaMemcpy(cpu_timer, thetimer, sizeof(clocktype) * 3 * fine_grid_expand_y.x, cudaMemcpyDeviceToHost));

    clocktype* cpu_timer2 = new clocktype[3 * coarse_grid_expand_y.x];
    memset(cpu_timer2, 0, 3 * coarse_grid_expand_y.x * sizeof(clocktype));
    checkCudaErrors(cudaMemcpy(cpu_timer2, thetimer2, sizeof(clocktype) * 3 * coarse_grid_expand_y.x, cudaMemcpyDeviceToHost));

    // analysis
    std::vector<Dur> v;
    std::vector<Dur> v2;
    for(int j = 0; j < fine_grid_expand_y.x; ++j)
    {
        v.push_back(Dur(cpu_timer[j * 3], cpu_timer[j * 3 + 1], (int)(cpu_timer[j * 3 + 2])));
    }

    for(int j = 0; j < coarse_grid_expand_y.x; ++j)
    {
        v2.push_back(Dur(cpu_timer2[j * 3], cpu_timer2[j * 3 + 1], (int)(cpu_timer2[j * 3 + 2])));
    }
    std::sort(v.begin(), v.end(), cmp); 
    std::sort(v2.begin(), v2.end(), cmp); 
    const int SMNUM = 80; // For V100
    int* runsm = new int[SMNUM];
    memset(runsm, 0, SMNUM * sizeof(int));
    c_begin = v[v.size() - 1].begin;
    int accu = 0;
    for(auto item : v)
    {
        int id = item.smid;
        if(runsm[id] == numBlocksPerSm)
        {
        continue;
        }
        else
        {
            runsm[id] ++;
            if(runsm[id] == numBlocksPerSm)
            {
                accu++;
                if(accu == SMNUM)
                {
                    break;
                }
            }
        }
    }

    double overall_time = 0;
    for(auto item : v)
    {
        overall_time += ((double)(item.end - item.begin));
    }
    overall_time /= 1e9;
    auto hkz_all_t0 = getDuration(t0, t1);
    dbg(hkz_all_t0);
    auto hkz_real_t0 = overall_time / (80 * numBlocksPerSm);
    dbg(hkz_real_t0);
    
    double overall_time2 = 0;
    for(auto item : v2)
    {
        overall_time2 += ((double)(item.end - item.begin));
    }
    overall_time2 /= 1e9;
    auto hkz_all_t1 = getDuration(t1, t2);
    dbg(hkz_all_t1);
    auto hkz_real_t1 = overall_time2 / (80 * numBlocksPerSm2);
    dbg(hkz_real_t1);

    FILE *fout0(fopen("results/fig8_base_actual.log", "a"));
    FILE *fout1(fopen("results/fig8_base_balanced.log", "a"));
    FILE *fout2(fopen("results/fig8_NG_actual.log", "a"));
    FILE *fout3(fopen("results/fig8_NG_balanced.log", "a"));
    assert(fout0 != NULL);
    fprintf(fout0, "%lf\n", hkz_all_t1);
    fprintf(fout1, "%lf\n", hkz_real_t1);
    fprintf(fout2, "%lf\n", hkz_all_t0);
    fprintf(fout3, "%lf\n", hkz_real_t0);
    fclose(fout0);
    fclose(fout1);
    fclose(fout2);
    fclose(fout3);
}
