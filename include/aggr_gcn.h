#ifndef AGGR_GCN_H
#define AGGR_GCN_H
#include "aggregator.h"

__global__ void aggr_gcn(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int col = (threadIdx.y << 5) + lane;
    if (row >= num_v)
        return;
    int begin = ptr[row], end = ptr[row + 1];
    float rs = 0.0f;
    int theidx;
    float theval;
    int jlimit;
#pragma unroll
    for (int i = begin; i < end; i += 32)
    {
        if (i + lane < end)
        {
            theidx = idx[i + lane] * INFEATURE;
            theval = val[i + lane];
        }
        jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
            rs += vin[__shfl(theidx, j, 32) + col] * __shfl(theval, j, 32);
        }
    }
    if (col < INFEATURE)
        vout[row * INFEATURE + col] = rs;
}

__global__ void aggr_gcn_shared(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    int col = (threadIdx.y << 5) + lane;

    extern __shared__ int sh_gcn[];
    int *shared_idx = (int *)(sh_gcn + warpid * 32);
    float *shared_val = (float *)(sh_gcn + blockDim.x + warpid * 32);

    int begin = ptr[row];
    int end = ptr[row + 1];
    int jlimit;

    if (col < INFEATURE)
    {
        float rs = 0.0f;
#pragma unroll
        for (int i = begin; i < end; i += 32)
        {
            if (i + lane < end)
            {
                shared_idx[lane] = idx[i + lane] * INFEATURE;
                shared_val[lane] = val[i + lane];
            }
            jlimit = 32;
            if (end - i < 32)
                jlimit = end - i;
            for (int j = 0; j < jlimit; ++j)
            {
                rs += vin[shared_idx[j] + col] * shared_val[j];
            }
        }
        vout[row * INFEATURE + col] = rs;
    }
}

__global__ void aggr_gcn_target(int *ptr, int *idx, float *val, int *targetv, float *vin, float *vout, int num_v, int INFEATURE, int NUM_OF_NEIGHBOR)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    int col = (threadIdx.y << 5) + lane;
    int whichv_fea = targetv[row] * INFEATURE;

    extern __shared__ int sh_gcn[];
    int *shared_idx = (int *)(sh_gcn + NUM_OF_NEIGHBOR * warpid);
    float *shared_val = (float *)(sh_gcn + NUM_OF_NEIGHBOR * (blockDim.x / 32) + NUM_OF_NEIGHBOR * warpid);

    int loc1 = ptr[row];
    int loc2 = ptr[row + 1];
    const int v_nei_num = loc2 - loc1;
#pragma unroll
    for (int i = lane; i < v_nei_num; i += 32)
    // for(int i = lane + 32 * threadIdx.y; i < v_nei_num; i+=32 * blockDim.y)
    {
        shared_idx[i] = idx[i + loc1] * INFEATURE;
        shared_val[i] = val[i + loc1];
    }
    // __syncthreads();

    if (col < INFEATURE)
    {
        float rs = 0.0f;
#pragma unroll
        for (int i = 0; i < v_nei_num; ++i)
        {
            rs += vin[shared_idx[i] + col] * shared_val[i];
        }
        atomicAdd(&vout[whichv_fea + col], rs);
    }
}

typedef uint64_t   clocktype ;
  struct Dur
  {
    clocktype begin;
    clocktype end;
    int smid = -1;
    Dur(clocktype x, clocktype y, int outsm)
    {
      begin = x;
      end = y;
      smid = outsm;
    }
  };
  bool cmp(Dur x, Dur y)
  {
    return (x.end > y.end);
  }
  static __device__ inline uint64_t GlobalTimer64(void) {
    // Due to a bug in CUDA's 64-bit globaltimer, the lower 32 bits can wrap
    // around after the upper bits have already been read. Work around this by
    // reading the high bits a second time. Use the second value to detect a
    // rollover, and set the lower bits of the 64-bit "timer reading" to 0, which
    // would be valid, it's passed over during the duration of the reading. If no
    // rollover occurred, just return the initial reading.
    volatile uint64_t first_reading;
    volatile uint32_t second_reading;
    uint32_t high_bits_first;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(first_reading));
    high_bits_first = first_reading >> 32;
    asm volatile("mov.u32 %0, %%globaltimer_hi;" : "=r"(second_reading));
    if (high_bits_first == second_reading) {
        return first_reading;
    }
    // Return the value with the updated high bits, but the low bits set to 0.
    return ((uint64_t) second_reading) << 32;
}
__device__ inline uint getSMId()
{
    uint smid;
    asm("mov.u32 %0, %smid;" : "=r" (smid));
    return smid;
}

__global__ void aggr_gcn_clock(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE, clocktype* timer)
{
    int smid = getSMId();
    clocktype tt = GlobalTimer64();

    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int col = (threadIdx.y << 5) + lane;
    if (row >= num_v)
        return;
    int begin = ptr[row], end = ptr[row + 1];
    float rs = 0.0f;
    int theidx;
    float theval;
    int jlimit;
#pragma unroll
    for (int i = begin; i < end; i += 32)
    {
        if (i + lane < end)
        {
            theidx = idx[i + lane] * INFEATURE;
            theval = val[i + lane];
        }
        jlimit = 32;
        if (end - i < 32)
            jlimit = end - i;
        for (int j = 0; j < jlimit; ++j)
        {
            rs += vin[__shfl(theidx, j, 32) + col] * __shfl(theval, j, 32);
        }
    }
    if (col < INFEATURE)
        vout[row * INFEATURE + col] = rs;

    clocktype tt2 = GlobalTimer64();
    if(threadIdx.x == 0)
    {
        timer[3 * blockIdx.x] = tt;
        timer[3 * blockIdx.x + 1] = tt2;
        timer[3 * blockIdx.x + 2] = (clocktype)(smid);
    }
}

__global__ void aggr_gcn_target_clock(int *ptr, int *idx, float *val, int *targetv, float *vin, float *vout, int num_v, int INFEATURE, int NUM_OF_NEIGHBOR, clocktype* timer)
{
    int smid = getSMId();
    clocktype tt = GlobalTimer64();

    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    int col = (threadIdx.y << 5) + lane;
    int whichv_fea = targetv[row] * INFEATURE;

    extern __shared__ int sh_gcn[];
    int *shared_idx = (int *)(sh_gcn + NUM_OF_NEIGHBOR * warpid);
    float *shared_val = (float *)(sh_gcn + NUM_OF_NEIGHBOR * (blockDim.x / 32) + NUM_OF_NEIGHBOR * warpid);

    int loc1 = ptr[row];
    int loc2 = ptr[row + 1];
    const int v_nei_num = loc2 - loc1;
#pragma unroll
    for (int i = lane; i < v_nei_num; i += 32)
    {
        shared_idx[i] = idx[i + loc1] * INFEATURE;
        shared_val[i] = val[i + loc1];
    }

    if (col < INFEATURE)
    {
        float rs = 0.0f;
#pragma unroll
        for (int i = 0; i < v_nei_num; ++i)
        {
            rs += vin[shared_idx[i] + col] * shared_val[i];
        }
        atomicAdd(&vout[whichv_fea + col], rs);
    }

    clocktype tt2 = GlobalTimer64();
    if(threadIdx.x == 0)
    {
        timer[3 * blockIdx.x] = tt;
        timer[3 * blockIdx.x + 1] = tt2;
        timer[3 * blockIdx.x + 2] = (clocktype)(smid);
    }
}

__global__ void aggr_gcn_target_shared(int *ptr, int *idx, float *val, int *targetv, float *vin, float *vout, int num_v, int INFEATURE)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (row >= num_v)
        return;
    int col = (threadIdx.y << 5) + lane;
    int whichv_fea = targetv[row] * INFEATURE;

    extern __shared__ int sh_gcn[];
    int *shared_idx = (int *)(sh_gcn + warpid * 32);
    float *shared_val = (float *)(sh_gcn + blockDim.x + warpid * 32);

    int begin = ptr[row];
    int end = ptr[row + 1];
    int jlimit;

    if (col < INFEATURE)
    {
        float rs = 0.0f;
#pragma unroll
        for (int i = begin; i < end; i += 32)
        {
            if (i + lane < end)
            {
                shared_idx[lane] = idx[i + lane] * INFEATURE;
                shared_val[lane] = val[i + lane];
            }
            jlimit = 32;
            if (end - i < 32)
                jlimit = end - i;
            for (int j = 0; j < jlimit; ++j)
            {
                rs += vin[shared_idx[j] + col] * shared_val[j];
            }
        }
        atomicAdd(&vout[whichv_fea + col], rs);
    }
}

__global__ void aggr_gcn_edgewise(int *edgelist, float *val, float *vin, float *vout, int num_e, int INFEATURE)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * (blockDim.x >> 5)) + warpid;
    if (row > num_e)
        return;
    int mysrc = edgelist[row * 2] * INFEATURE + lane;
    int mydst = edgelist[row * 2 + 1] * INFEATURE + lane;
    float myval = val[row];
    atomicAdd(&vout[mydst], vin[mysrc] * myval);
}

__global__ void aggr_gcn_nn(int* ptr, int* idx, int* targetv, float* val, float* vin, float* weight, float* vout, float* transformed, int num_v, int INFEATURE, int OUTFEATURE, int NEIGHBOR_NUM)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * blockDim.x / 32) + (threadIdx.x >> 5);
    if(row >= num_v) return;
    int col = (threadIdx.y << 5) + lane;
    int whichv = targetv[row];
    // __shared__ int sh[NEIGHBOR_NUM * 2 * TARGET_IN_BLOCK];
    extern __shared__ float ss[]; // weight for dense
    int* shared_idx = (int*)(ss + NEIGHBOR_NUM * warpid);
    float* shared_val = (float*)(ss + NEIGHBOR_NUM * blockDim.x / 32 + NEIGHBOR_NUM * warpid);
    float* shared_weight = (float*)(ss + NEIGHBOR_NUM * blockDim.x / 32 * 2) + threadIdx.y * 32 * OUTFEATURE;

    int loc1 = ptr[row];
    int loc2 = ptr[row + 1];
    const int v_nei_num = loc2 - loc1;

    #pragma unroll
    for(int i = lane; i < v_nei_num; i+=32)
    // for(int i = lane + 32 * threadIdx.y; i < v_nei_num; i+=32 * blockDim.y)
    {
        shared_idx[i] = idx[i + loc1] * INFEATURE;
        shared_val[i] = val[i + loc1];
    }
    // __syncthreads();

    float rs = 0.0f;
    if(col < INFEATURE)
    {
        #pragma unroll
        for(int i = 0; i < v_nei_num; ++i)
        {
            rs += vin[shared_idx[i] + col] * shared_val[i];
        }
        atomicAdd(&vout[whichv * INFEATURE + col], rs);
    }
    const int weight_base = threadIdx.y * 32 * OUTFEATURE;
    for(int i = threadIdx.x; i < 32 * OUTFEATURE; i += blockDim.x)
    {
        shared_weight[i] = weight[i + weight_base];
    }
    __syncthreads();
    float ans = 0;
    // for(int i = threadIdx.y * 32; i < (threadIdx.y + 1)*32; ++i)
    // {
    //     ans += shared_weight[i * OUTFEATURE + lane] * __shfl(rs, (i)%32, 32);
    // }
    for(int i = 0; i < 32; ++i)
    {
        ans += shared_weight[i * OUTFEATURE + lane] * __shfl(rs, i, 32);
    }
    if(lane < OUTFEATURE)
        atomicAdd(&transformed[whichv * OUTFEATURE + lane], ans);
        // transformed[row * OUTFEATURE + col] = ans;
}


class Aggregator_GCN : public Aggregator
{
public:
    Aggregator_GCN(int *host_out_ptr, int *host_out_idx, int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in, int out_feat_out, float *out_val) : Aggregator(host_out_ptr, host_out_idx, dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in, out_feat_out), d_val(out_val)
    {
        h_val = new float[num_e];
        checkCudaErrors(cudaMemcpy(h_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost));
    }
    Aggregator_GCN(CSRSubGraph g, int out_feat_in, int out_feat_out, float *out_val) : Aggregator(g, out_feat_in, out_feat_out), d_val(out_val)
    {
        h_val = new float[num_e];
        checkCudaErrors(cudaMemcpy(h_val, d_val, num_e * sizeof(float), cudaMemcpyDeviceToHost));
    }
    ~Aggregator_GCN()
    {
        safeFree(d_val);
    }
    double run(float *vin, float *vout, int BLOCK_SIZE, bool scheduled) override
    {
        int tmp_target_in_block = BLOCK_SIZE / feat_in;

        int shared_size = 0;

        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);

        if (scheduled)
        {
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            shared_size = neighbor_group_size * 2 * tmp_target_in_block * sizeof(int);
            assert(d_ptr_scheduled != NULL);
            checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
        }

        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t0);
        if (scheduled)
        {
            aggr_gcn_target<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_val_scheduled, d_target_scheduled, vin, vout, num_target, feat_in, neighbor_group_size);
        }
        else
        {
            aggr_gcn<<<grid, block, shared_size>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
            // aggr_gcn_shared<<<grid, block, BLOCK_SIZE * 2 * sizeof(float)>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t1);
        return 0.0;
    }
    double run_with_feat(float *vin, float *vout, int BLOCK_SIZE, bool scheduled, int feat) 
    {
        feat_in = feat;
        if (BLOCK_SIZE < feat_in) BLOCK_SIZE = feat_in;
        int tmp_target_in_block = BLOCK_SIZE / feat_in;

        int shared_size = 0;

        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);

        if (scheduled)
        {
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            shared_size = neighbor_group_size * 2 * tmp_target_in_block * sizeof(int);
            assert(d_ptr_scheduled != NULL);
            checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
        }

        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t0);
        if (scheduled)
        {
            aggr_gcn_target<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_val_scheduled, d_target_scheduled, vin, vout, num_target, feat_in, neighbor_group_size);
        }
        else
        {
            aggr_gcn<<<grid, block, shared_size>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
            // aggr_gcn_shared<<<grid, block, BLOCK_SIZE * 2 * sizeof(float)>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t1);
        return 0.0;
    }
    double runEdgeWise(float *vin, float *vout, int BLOCK_SIZE, bool scheduled) override
    {
        checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
        int tmp_target_in_block = BLOCK_SIZE / 32;
        dim3 grid((num_e + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE);
        if (d_edgelist == NULL)
            csr2edgelist();

        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t0);
        aggr_gcn_edgewise<<<grid, block>>>(d_edgelist, d_val, vin, vout, num_e, feat_in);
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        return getDuration(t0, t1);
    }

    double run_clock(float *vin, float *vout, clocktype* timer, int BLOCK_SIZE, bool scheduled)
    {
        int tmp_target_in_block = BLOCK_SIZE / feat_in;
        int shared_size = 0;
        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);
        if (scheduled)
        {
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            shared_size = neighbor_group_size * 2 * tmp_target_in_block * sizeof(int);
            assert(d_ptr_scheduled != NULL);
            checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
        }

        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t0);
        if (scheduled)
        {
            aggr_gcn_target_clock<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_val_scheduled, d_target_scheduled, vin, vout, num_target, feat_in, neighbor_group_size, timer);
        }
        else
        {
            aggr_gcn_clock<<<grid, block, shared_size>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in, timer);
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        return getDuration(t0, t1);
    }

    void run_with_nn(float* vin, float* vout, float* weight, float* transformed, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / feat_in;
        dim3 grid((num_target + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);
        size_t shared_size = (neighbor_group_size * tmp_target_in_block * 2 + feat_in * feat_out) * sizeof(int);
        aggr_gcn_nn <<< grid, block, shared_size >>>
            (d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, d_val_scheduled, vin, weight, vout, transformed, num_target, feat_in, feat_out, neighbor_group_size);
    }

    void schedule(Schedule s, int *param) override
    {
        if (s == neighbor_grouping)
        {
            Aggregator::schedule(s, param);
            d_val_scheduled = d_val;
            return;
        }
        assert(s == locality || s == locality_neighbor_grouping);
        std::vector<int> ptr_vec;
        std::vector<int> idx_vec;
        std::vector<int> target_vec;
        std::vector<float> val_vec;
        sche = s;
        safeFree(d_ptr_scheduled);
        safeFree(d_idx_scheduled);
        safeFree(d_target_scheduled);
        safeFree(d_val_scheduled);
        switch (s)
        {
        case locality:
            locality_schedule(h_ptr, h_idx, param[0], num_v, &ptr_vec, &idx_vec, &target_vec, n, h_val, &val_vec);
            locality_partition_num = param[0];
            break;
        case locality_neighbor_grouping:
            localityNeighborGrouping(h_ptr, h_idx, param[0], param[1], num_v, &ptr_vec, &idx_vec, &target_vec, n, h_val, &val_vec);
            locality_partition_num = param[0];
            neighbor_group_size = param[1];
            break;
        default:
            break;
        }
        num_target = target_vec.size();
        copyVec2Dev(&ptr_vec, d_ptr_scheduled);
        copyVec2Dev(&idx_vec, d_idx_scheduled);
        copyVec2Dev(&val_vec, d_val_scheduled);
        copyVec2Dev(&target_vec, d_target_scheduled);
    }

    void updateval(float* out_d_val)
    {
        d_val_scheduled = out_d_val;
        d_val = out_d_val;
    }

private:
    float *d_val = NULL;
    float *h_val = NULL;
    float *d_val_scheduled = NULL;
};

#endif
