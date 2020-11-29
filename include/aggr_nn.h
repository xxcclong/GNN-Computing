#ifndef AGGR_NN_H
#define AGGR_NN_H
#include "aggregator.h"

#define TILING 3
// #define TIMING 2
#define USE_CONSTANT

// __constant__ float const_weight[32 * 32];

#define COMP                                                                   \
    do                                                                         \
    {                                                                          \
        extern __shared__ float sh_mlp[];                                      \
        float *shared_weight = sh_mlp;                                         \
        int *shared_idx = (int *)(sh_mlp + 32 * 32 + warpid * 32);             \
        float *input = (float *)(sh_mlp + 32 * 32 + blockDim.x + warpid * 32); \
        const int weight_size = 32 * 32;                                       \
        for (int i = 0; i < weight_size; i += blockDim.x)                      \
        {                                                                      \
            if (threadIdx.x + i < weight_size)                                 \
                shared_weight[i + threadIdx.x] = weight[i + threadIdx.x];      \
        }                                                                      \
        __syncthreads();                                                       \
                                                                               \
        float cached = vin[whichv_fea + lane];                                 \
                                                                               \
        for (int i = begin; i < end; i += 32)                                  \
        {                                                                      \
            shared_idx[lane] = idx[i + lane] * 32;                             \
            int jlimit = 32;                                                   \
            if (end - i < 32)                                                  \
                jlimit = end - i;                                              \
            for (int j = 0; j < jlimit; ++j)                                   \
            {                                                                  \
                input[lane] = cached + vin[shared_idx[j] + lane];              \
                __syncwarp();                                                  \
                float ans = 0;                                                 \
                for (int k = 0; k < 32; ++k)                                   \
                {                                                              \
                    ans += input[k] * shared_weight[lane + 32 * k];            \
                }                                                              \
                if (ans > 0)                                                   \
                    rs += ans;                                                 \
            }                                                                  \
        }                                                                      \
    } while (0)

#define PURE_COMP

__global__
#ifdef TIMING
    void
    aggr_mlp(const int *ptr, const int *idx, const float *val, const float *vin, float *vout, const float *weight, const int num_v, const int INFEATURE, const int OUTFEATURE, clocktype *timer)
#else
    void
    aggr_mlp(const int *ptr, const int *idx, const float *val, const float *vin, float *vout, const float *weight, const int num_v, const int INFEATURE, const int OUTFEATURE)
#endif
{
#ifdef TIMING
    int smid = getSMId();
    clocktype tt, tt2;
#endif

#if TIMING == 2
    clocktype total_time = 0;
#endif

#if TIMING == 1
    tt = GlobalTimer64();
#endif
    const int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    // const int col = (threadIdx.y << 5) + lane;
    const int begin = ptr[row], end = ptr[row + 1], v_nei_num = end - begin;
    const int whichv_fea = row * 32;
    float rs = 0;

#ifdef PURE_COMP
    COMP;
    vout[whichv_fea + lane] = rs;
#else
    extern __shared__ float sh_mlp[];
    float *shared_weight = sh;                                              // 32 * 32
    int *shared_idx = (int *)(sh_mlp + 32 * 32 + warpid * 32);              // 32 for each warp, 32 * warpid
#if TILING == 2
    float *output = (float *)(sh_mlp + 32 * 32 + blockDim.x + warpid * 32); // 32
#elif TILING == 3
    float *input = (float *)(sh_mlp + 32 * 32 + blockDim.x + warpid * 32); // 32
#endif

    // load weight (32 * 32)
    const int weight_size = 32 * 32;
    for (int i = 0; i < weight_size; i += blockDim.x)
    {
        if (threadIdx.x + i < weight_size)
            shared_weight[i + threadIdx.x] = weight[i + threadIdx.x];
    }
    __syncthreads();

    // cache dst feat
    float cached = vin[whichv_fea + lane];

    // computation
    // assuming input == output == 32
    float rs = 0;
    for (int i = begin; i < end; i += 32)
    {
        shared_idx[lane] = idx[i + lane] * 32;
        // shared_idx[lane] = 0;//(i%10) * 32;//idx[i + lane] * 32;
        int jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
#if TILING == 1
            // tile type 1
            float res = cached + vin[shared_idx[j] + lane];
            float ans = 0;
            for (int k = 0; k < 32; ++k)
            {
                // ans += __shfl(res, k, 32) * weight[lane + 32 * k];
                ans += __shfl(res, k, 32) * shared_weight[lane + 32 * k];
                // ans += __shfl(res, k, 32) * scalar;
                // ans += __shfl(res, k, 32) * 5.5f;
            }
            if (ans > 0) // relu
                rs += ans;
#elif TILING == 2
            // tile type 2
            // allocate new shared memory for output
            float res = cached + vin[shared_idx[j] + lane];
            output[lane] = 0;
            __syncwarp();
            for (int k = 0; k < 32; ++k)
            {
                // atomicAdd(&output[k], res * shared_weight[k * 32 + lane]);
                atomicAdd(&output[k], res * shared_weight[k + lane * 32]);
            }
            __syncwarp();
            if (output[lane] > 0)
                rs += output[lane];
#elif TILING == 3
            // alocate shared memory for mid res
            input[lane] = cached + vin[shared_idx[j] + lane];
            __syncwarp();
            float ans = 0;
#if TIMING == 2
            __syncthreads();
            tt = GlobalTimer64();
#endif
#pragma unroll
            for (int k = 0; k < 32; ++k)
            // for(int k = 0; k < 2; ++k)
            {
                // ans += rs * rs; //* shared_weight[lane + 32 * k];
                // ans += input[k] * shared_weight[lane * 32 + k];
                ans += input[k] * shared_weight[lane + 32 * k];
                // ans += input[k] * const_weight[lane + 32 * k];
            }
#if TIMING == 2
            __syncthreads();
            tt2 = GlobalTimer64();
            total_time += (tt2 - tt);
#endif
            if (ans > 0) // relu
                rs += ans;
#endif
        }
    }
    vout[whichv_fea + lane] = rs;

#if TIMING == 1
    tt2 = GlobalTimer64();
#endif

#if TIMING == 1
    if (threadIdx.x == 0)
    {
        timer[3 * blockIdx.x] = tt;
        timer[3 * blockIdx.x + 1] = tt2;
        timer[3 * blockIdx.x + 2] = (uint64_t)(smid);
    }
#elif TIMING == 2
    if (threadIdx.x == 0)
    {
        timer[3 * blockIdx.x] = 0;
        timer[3 * blockIdx.x + 1] = total_time;
        timer[3 * blockIdx.x + 2] = (uint64_t)(smid);
    }
#endif
#endif
}

__global__ void aggr_mlp_target(int *ptr, int *idx, int *target, float *vin, float *vout, float *weight, int num_v, int INFEATURE, int OUTFEATURE)
{
    int which = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (which >= num_v)
        return;
    const int row = target[which];
    const int lane = threadIdx.x & 31;
    const int warpid = threadIdx.x >> 5;
    // const int col = (threadIdx.y << 5) + lane;
    const int begin = ptr[which], end = ptr[which + 1], v_nei_num = end - begin;
    const int whichv_fea = row * 32;
    float rs = 0;

#ifdef PURE_COMP
    COMP;
    atomicAdd(&vout[whichv_fea + lane], rs);
#else
    // a warp in charge of a center vertex
    extern __shared__ float sh_mlp[];
    float *shared_weight = sh;                                              // 32 * 32
    int *shared_idx = (int *)(sh_mlp + 32 * 32 + warpid * 32);              // 32 for each warp, 32 * warpid
#if TILING == 2
    float *output = (float *)(sh_mlp + 32 * 32 + blockDim.x + warpid * 32); // 32
#elif TILING == 3
    float *input = (float *)(sh_mlp + 32 * 32 + blockDim.x + warpid * 32); // 32
#endif

    // load weight (32 * 32)
    const int weight_size = 32 * 32;
    for (int i = 0; i < weight_size; i += blockDim.x)
    {
        if (threadIdx.x + i < weight_size)
            shared_weight[i + threadIdx.x] = weight[i + threadIdx.x];
    }

    // cache dst feat
    float cached = vin[whichv_fea + lane];
    __syncthreads();

    // computation
    // assuming input == output == 32
    float rs = 0;
    for (int i = begin; i < end; i += 32)
    {
        shared_idx[lane] = idx[i + lane] * 32;
        int jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
            float res = cached + vin[shared_idx[j] + lane];
#if TILING == 1
            // tile type 1
            float ans = 0;
            for (int k = 0; k < 32; ++k)
            {
                // ans += (cached_dst_feat[k] + vin[shared_idx[j] + k]) * weight[lane + 32 * k];
                ans += __shfl(res, k, 32) * shared_weight[lane + 32 * k];
            }
            if (ans > 0) // relu
                rs += ans;
#elif TILING == 2
            // tile type 2
            // allocate new shared memory for output
            output[lane] = 0;
            __syncwarp();
            for (int k = 0; k < 32; ++k)
            {
                atomicAdd(&output[k], res * shared_weight[k + lane * 32]);
            }
            __syncwarp();
            if (output[lane] > 0)
                rs += output[lane];
#elif TILING == 3
            // alocate shared memory for mid res
            input[lane] = res;
            __syncwarp();
            float ans = 0;
            for (int k = 0; k < 32; ++k)
            {
                // ans += (cached_dst_feat[k] + vin[shared_idx[j] + k]) * weight[lane + 32 * k];
                ans += input[k] * shared_weight[lane + 32 * k];
            }
            if (ans > 0) // relu
                rs += ans;
#endif
        }
    }
    atomicAdd(&vout[whichv_fea + lane], rs);
#endif
}

class Aggregator_MLP : public Aggregator
{
public:
    Aggregator_MLP(int *host_out_ptr, int *host_out_idx, int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in, int out_feat_out, float *out_weight) : Aggregator(host_out_ptr, host_out_idx, dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in, out_feat_out), d_weight(out_weight)
    {
    }
    Aggregator_MLP(CSRSubGraph g, int out_feat_in, int out_feat_out, float *out_weight) : Aggregator(g, out_feat_in, out_feat_out), d_weight(out_weight) {}
    double run(float *vin, float *vout, int BLOCK_SIZE, bool scheduled) override
    {
        int tmp_target_in_block = BLOCK_SIZE / feat_in;
#if TILING == 1
        int shared_size = (BLOCK_SIZE + feat_in * feat_in) * sizeof(float);
#elif TILING == 2
        int shared_size = (BLOCK_SIZE + feat_in * feat_in + feat_in * tmp_target_in_block) * sizeof(float);
#elif TILING == 3
        int shared_size = (BLOCK_SIZE + feat_in * feat_in + feat_in * tmp_target_in_block) * sizeof(float);
#else
        int shared_size = 0;
#endif
        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE);

        if (scheduled)
        {
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            assert(d_ptr_scheduled != NULL);
            checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
        }

        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t0);
        if (scheduled)
        {
            aggr_mlp_target<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, vin, vout, d_weight, num_target, feat_in, feat_out);
        }
        else
        {
#ifdef TIMING
            assert(false);
            aggr_mlp<<<coarse_grid_expand_y, coarse_block_expand_y, shared_size>>>(d_ptr, d_idx, NULL, vin, vout, d_weight, num_v, feat_in, feat_out, thetimer);
#else
            aggr_mlp<<<grid, block, shared_size>>>(d_ptr, d_idx, NULL, vin, vout, d_weight, num_v, feat_in, feat_out);
#endif
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        return getDuration(t0, t1);
    }

private:
    float *d_weight;
};

#endif
