#ifndef AGGR_SDDMM_H
#define AGGR_SDDMM_H
#include "aggregator.h"

__global__ void aggr_sddmm(const int *ptr, const int *idx, float *val, const float *vin1, const float *vin2, const int num_v, const int INFEATURE)
{
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    const int begin = ptr[row], end = ptr[row + 1], v_nei_num = end - begin;
    extern __shared__ int sh[];
    int *shared_idx = sh + warpid * 32;
    float *shared_write_cache = (float *)(sh + blockDim.x + warpid * 32);

    float cached = vin2[row * INFEATURE + lane];

    for (int i = begin; i < end; i += 32)
    {
        shared_idx[lane] = idx[i + lane] * 32;
        int jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
            const int pos = shared_idx[j] + lane;
            float res = vin1[pos] * cached;
            for (int k = 16; k > 0; k >>= 1)
            {
                res += __shfl_down_sync(0xffffffff, res, k);
            }
            if (lane == 0)
            {
                // val[i + j] = res;
                shared_write_cache[j] = res;
            }
        }
        // __syncwarp();
        if (i + lane < end)
            val[i + lane] = shared_write_cache[lane];
    }
}

__global__ void aggr_sddmm_target(const int *ptr, const int *idx, int *target, float *val, const float *vin1, const float *vin2, const int num_v, const int INFEATURE)
{
    int which = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (which >= num_v)
        return;
    const int row = target[which];
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    const int begin = ptr[which], end = ptr[which + 1], v_nei_num = end - begin;
    extern __shared__ int sh[];
    int *shared_idx = sh + warpid * 32;
    float *shared_write_cache = (float *)(sh + blockDim.x + warpid * 32);

    float cached = vin2[row * INFEATURE + lane];
    for (int i = begin; i < end; i += 32)
    {
        shared_idx[lane] = idx[i + lane] * 32;
        int jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
            const int pos = shared_idx[j] + lane;
            float res = vin1[pos] * cached;
            for (int k = 16; k > 0; k >>= 1)
            {
                res += __shfl_down_sync(0xffffffff, res, k);
            }
            if (lane == 0)
            {
                // val[i + j] = res;
                shared_write_cache[j] = res;
            }
        }
        // __syncwarp();
        if (i + lane < end)
            val[i + lane] = shared_write_cache[lane];
    }
}

class Aggregator_SDDMM : public Aggregator
{
public:
    Aggregator_SDDMM(int *host_out_ptr, int *host_out_idx, int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in, int out_feat_out) : Aggregator(host_out_ptr, host_out_idx, dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in, out_feat_out)
    {
    }
    Aggregator_SDDMM(CSRSubGraph g, int out_feat_in, int out_feat_out) : Aggregator(g, out_feat_in, out_feat_out) {}
    double run(float *v1, float *v2, float *outval, int BLOCK_SIZE, bool scheduled) override
    {
        int tmp_target_in_block = BLOCK_SIZE / 32;
        int shared_size = BLOCK_SIZE * 2 * sizeof(float);
        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE);
        if (scheduled)
        {
            assert(sche == neighbor_grouping);
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            assert(d_ptr_scheduled != NULL);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t0);
        if (scheduled)
        {
            aggr_sddmm_target<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, outval, v1, v2, num_target, feat_in);
        }
        else
        {
            aggr_sddmm<<<grid, block, shared_size>>>(d_ptr, d_idx, outval, v1, v2, num_v, feat_in);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        return getDuration(t0, t1);
    }

private:
};

#endif