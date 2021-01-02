#ifndef AGGR_GAT_H
#define AGGR_GAT_H
#include "aggregator.h"

__global__ void attGat(int* ptr, int* idx, float* newval, float* transform, int num_v, float relu_l)
{
    int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int laneid = threadIdx.x & 31;
    if(warpid < num_v)
    {
        float partial_sum = 0;
        float local_v = transform[warpid << 1];
        int begin = ptr[warpid], end = ptr[warpid + 1];
        for(int i = begin + laneid; i < end; i += 32)
        {
            float tmpsum = local_v + transform[(idx[i] << 1) + 1];
            tmpsum = __expf(max(tmpsum, tmpsum * relu_l));
            newval[i] = tmpsum;
            partial_sum += tmpsum;
        }
        for(int i = 16; i > 0; i >>= 1)
        {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i);
        }
        partial_sum = __shfl_sync(0xffffffff, partial_sum, 0);
        for(int i = begin + laneid; i < end; i += 32)
        {
            newval[i] /= partial_sum;
        }
    }
}

__global__ void u_add_v(int* ptr, int* idx, float* newval, float* transform, int num_v)
{
    int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int laneid = threadIdx.x & 31;

    if(warpid < num_v)
    {
        float partial_sum = 0;
        float local_v = transform[warpid << 1];
        int begin = ptr[warpid], end = ptr[warpid + 1];
        for(int i = begin + laneid; i < end; i += 32)
        {
            newval[i] = local_v + transform[(idx[i] << 1) + 1];
        }
    }
}

__global__ void add_to_center(int* ptr, int* idx, float* newval, float* transform, int num_v)
{
    int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int laneid = threadIdx.x & 31;

    if(warpid < num_v)
    {
        float partial_sum = 0;
        //float local_v = transform[warpid << 1];
        int begin = ptr[warpid], end = ptr[warpid + 1];
        for(int i = begin + laneid; i < end; i += 32)
        {
            //newval[i] = local_v + transform[(idx[i] << 1) + 1];
            partial_sum += newval[i];
        }
        for(int i = 16; i > 0; i >>= 1)
        {
            partial_sum += __shfl_down_sync(0xffffffff, partial_sum, i);
        }
        if(laneid == 0)
        {
            transform[warpid] = partial_sum;
        }
    }
}

__global__ void each_div(int* ptr, int* idx, float* newval, float* transform, int num_v)
{
    int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    int laneid = threadIdx.x & 31;

    if(warpid < num_v)
    {
        float local_v = transform[warpid];
        int begin = ptr[warpid], end = ptr[warpid + 1];
        for(int i = begin + laneid; i < end; i += 32)
        {
            //newval[i] = local_v + transform[(idx[i] << 1) + 1];
            //partial_sum += newval[i];
            newval[i] /= local_v;
        }
    }
}

// __global__ void each_div_bwd(int* ptr, int* idx, float* output, float* doutput, float* numerator, float* denominator, /*output gradient*/float* d_nominator, float* d_denominator, int num_v)
// {
//     int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
//     int laneid = threadIdx.x & 31;
//     if(warpid < num_v)
//     {
//         int begin = ptr[warpid], end = ptr[warpid + 1];
//         float accumulate = 0.f;
//         float cached_deno = denominator[warpid];
//         for(int i = begin + laneid; i < end; i += 32)
//         {
//             accumulate += output[i] * (-1) * cached_deno;
//             d_nominator[i] = cached_deno;
//         }
//         for (int k = 16; k > 0; k >>= 1)
//         {
//             accumulate += __shfl_down_sync(0xffffffff, accumulate, k);
//         }
//     }
// }


__global__ void aggr_gat(int* ptr, int* idx, float* newval, float* vin, float* vout, float* transform, int num_v, int INFEATURE, float relu_l)
{
    const int TARGET_IN_BLOCK = blockDim.x >> 5;
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * TARGET_IN_BLOCK) + (threadIdx.x >> 5);
    int col = (threadIdx.y << 5) + lane;

    if(row >= num_v) return;
    const float local_v = transform[row << 1];
    int begin = ptr[row], end = ptr[row + 1];
    float rs = 0.0f;
    int theidx;
    float theval;
    int jlimit;
    float partial_sum = 0;
    #pragma unroll
    for(int i = begin; i < end; i += 32)
    {
        if(i + lane < end)
        {
            theidx = idx[i + lane];
            theval = local_v + transform[(theidx << 1) + 1];
            theidx *= INFEATURE;
            // theval = max(theval, theval * relu_l);
            // theval = __nv_max(theval, theval * 0.1);
            // if(theval < 0) theval *= 0.1;
            theval = __expf(max(theval, theval * relu_l));
            // newval[i + lane] = theval;
        }
        jlimit = 32;
        if(end - i < 32) jlimit = end - i;
        for(int j = 0; j < jlimit; ++j)
        {
            float tmpval = __shfl(theval, j, 32);
            rs += vin[__shfl(theidx, j, 32) + col] * tmpval;
            partial_sum += tmpval;

        }
    }

    // for(int i = begin + lane; i < end; i += 32)
    // {
    //     newval[i] /= partial_sum;
    // }

    if(col < INFEATURE)
        vout[row * INFEATURE + col] = rs / partial_sum;
}


__global__ void aggr_gat_fine(int* ptr, int* idx, int* targetv, float* newval, float* vin, float* vout, float* scalar, float* transform, int num_v, int INFEATURE, int NEIGHBOR_NUM, float relu_l)
{
    const int TARGET_IN_BLOCK = blockDim.x / 32;
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * TARGET_IN_BLOCK) + (threadIdx.x >> 5);
    if(row >= num_v) return;
    int col = (threadIdx.y << 5) + lane;
    int which_v = targetv[row];
    // __shared__ int sh[NEIGHBOR_NUM * 2 * TARGET_IN_BLOCK];
    extern __shared__ int sh_gat[];
    int* shared_idx = (int*)(sh_gat + NEIGHBOR_NUM * warpid);
    float* shared_val = (float*)(sh_gat + NEIGHBOR_NUM * TARGET_IN_BLOCK + NEIGHBOR_NUM * warpid);

    int begin = ptr[row];
    int end = ptr[row + 1];
    const int v_nei_num = end - begin;
    const float local_v = transform[which_v << 1];
#pragma unroll
    for(int i = lane; i < v_nei_num; i += 32)
    {
        shared_idx[i] = idx[i + begin] * INFEATURE;
        float tmpsum = local_v + transform[(idx[i + begin] << 1) + 1];
        tmpsum = __expf(max(tmpsum, tmpsum * relu_l));
        shared_val[i] = tmpsum;
        newval[begin + i] = tmpsum;
    }
    float rs = 0.0f, partial_sum = 0.0f;
    #pragma unroll
    for(int i = 0; i < v_nei_num; ++i)
    {
        rs += vin[shared_idx[i] + col] * shared_val[i];
        partial_sum += shared_val[i];
    }
    if(col < INFEATURE)
        atomicAdd(&vout[which_v * INFEATURE + col], rs);
    if(lane == 0 && col == 0)
        atomicAdd(&scalar[which_v], partial_sum);
}

__global__ void scaleArray(float* vin, float* scalar, int INFEATURE, int num_v)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float div = 0;
    if((tid < num_v * INFEATURE) && (div = scalar[tid / INFEATURE]) != 0)
        vin[tid] /= div;
}

//////////////////////////////////
// Begin Experiment
//////////////////////////////////
// __global__ scaleArrayBwd()
// {

// }
__global__ void aggr_gat_fine_bwd(int* ptr, int* idx, int* targetv, float* output, float* doutput, float* newval/*exp(relu(a+b))*/, /*float* addval // before relu*/ float* div, float* infeat, /*float* a_b (a+b) on_edge*/ /*outputs*/ float* d_a_b, float* d_feat, float relu_l, int num_v, int INFEATURE, int NEIGHBOR_NUM)
{
    const int TARGET_IN_BLOCK = blockDim.x / 32;
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int row = (blockIdx.x * TARGET_IN_BLOCK) + (threadIdx.x >> 5);
    if(row >= num_v) return;
    int col = (threadIdx.y << 5) + lane;
    int which_v = targetv[row];
    extern __shared__ int sh_gat[];

    int* shared_idx = (int*)(sh_gat + NEIGHBOR_NUM * warpid);
    float* shared_val = (float*)(sh_gat + NEIGHBOR_NUM * TARGET_IN_BLOCK + NEIGHBOR_NUM * warpid);
    float* shared_write_cache = (float*)(sh_gat + NEIGHBOR_NUM * TARGET_IN_BLOCK * 2 + warpid * NEIGHBOR_NUM);


    int begin = ptr[row];
    int end = ptr[row + 1];
    const int v_nei_num = end - begin;
    const float thediv = div[which_v];
    const float cached_d_output = doutput[which_v * INFEATURE + lane] / thediv; // NEIGHBOR_NUM

#pragma unroll
    for (int i = lane; i < v_nei_num; i += 32)
    {
        shared_idx[i] = idx[i + begin];
        shared_val[i] = newval[i + begin];
    }

    // get d_feat
    // Adj_T * doutput
    // SpMM
    // for(int i = 0; i < v_nei_num; ++i)
    // {
    // }

    // get d_newval
    // feat * doutput_T
    // SDDMM
    for(int i = 0; i < v_nei_num; ++i)
    {
        atomicAdd(&(d_feat[shared_idx[i] * INFEATURE + col]), cached_d_output * shared_val[i]);
        float res = infeat[shared_idx[i] * INFEATURE + col] * cached_d_output;
        for (int k = 16; k > 0; k >>= 1)
        {
            res += __shfl_down_sync(0xffffffff, res, k);
        }
        if (lane == 0)
        {
            shared_write_cache[i] = res;
        }
    }

    // get d_(reducesum(exp(relu(a+b))))
    // sum(d_output * output) * (-1) / (div^2)
    // = sum(d_output * (output / div)) / -div
    // sum up 
    const int pos = which_v * INFEATURE + col;
    float res = doutput[pos] * output[pos] * (-1) / thediv;
    for (int k = 16; k > 0; k >>= 1)
    {
        res += __shfl_down_sync(0xffffffff, res, k);
    }
    res = __shfl_sync(0xffffffff, res, 0);
        

    for(int i = lane; i < v_nei_num; i += 32)
    {
        float thediff = (res + shared_write_cache[i]) * newval[begin + i];
        if(newval[begin + i] < 0) thediff *= relu_l; 
        atomicAdd(&(d_a_b[shared_idx[i] * 2 + 1]), thediff);
    }
}
//////////////////////////////////
// End Experiment
//////////////////////////////////

class Aggregator_GAT : public Aggregator
{
public:
    Aggregator_GAT(int *host_out_ptr, int *host_out_idx, int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in, int out_feat_out) : Aggregator(host_out_ptr, host_out_idx, dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in, out_feat_out)
    {
        checkCudaErrors(cudaMalloc2((void**)&scalar, num_v * sizeof(float)));
        checkCudaErrors(cudaMemset(scalar, 0, num_v * sizeof(float)));
        checkCudaErrors(cudaMalloc2((void**)&d_newval, num_e * sizeof(float)));
    }
    Aggregator_GAT(CSRSubGraph g, int out_feat_in, int out_feat_out) : Aggregator(g, out_feat_in, out_feat_out)
    {
        checkCudaErrors(cudaMalloc2((void**)&scalar, num_v * sizeof(float)));
        checkCudaErrors(cudaMemset(scalar, 0, num_v * sizeof(float)));
        checkCudaErrors(cudaMalloc2((void**)&d_newval, num_e * sizeof(float)));
    }
    ~Aggregator_GAT()
    {
    }
    double run(float *vin, float *vatt, float *vout, int BLOCK_SIZE, bool scheduled) override
    {
        int tmp_target_in_block = BLOCK_SIZE / feat_in;

        int shared_size = 0;

        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);
        // float* newval = NULL;

        if (scheduled)
        {
            // dbg(num_target);
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            shared_size = neighbor_group_size * 2 * tmp_target_in_block * sizeof(int);
            // assert(d_ptr_scheduled != NULL);
            // checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
            // checkCudaErrors(cudaMalloc2((void**)&newval, num_e * sizeof(int)));
        }

        if (scheduled)
        {
            aggr_gat_fine<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, /*newval*/ d_newval, vin, vout, scalar, vatt,  num_target, feat_in, neighbor_group_size, 0.2f);
            scaleArray <<< (num_v * feat_in + 256 - 1) / 256, 256 >>>
                (vout, scalar, feat_in, num_v);
            // testGPUBuffer(0, scalar);
            // testGPUBuffer(0, d_newval);
        }
        else
        {
            aggr_gat<<<grid, block, shared_size>>>(d_ptr, d_idx, d_val, vin, vout, vatt, num_v, feat_in, 0.2f);
        }
        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t1);
        // dbg(getDuration(t0, t1));
        // return getDuration(t0, t1);
        return 0.0;
    }
    double run_with_feat(float *vin, float *vatt, float *vout, int BLOCK_SIZE, bool scheduled, int feat)
    {
        feat_in = feat;
        if (BLOCK_SIZE < feat_in) BLOCK_SIZE = feat_in;
        int tmp_target_in_block = BLOCK_SIZE / feat_in;

        int shared_size = 0;

        dim3 grid((num_v + tmp_target_in_block - 1) / tmp_target_in_block);
        dim3 block(BLOCK_SIZE / (feat_in / 32), feat_in / 32);
        // float* newval = NULL;

        if (scheduled)
        {
            // dbg(num_target);
            grid.x = (num_target + tmp_target_in_block - 1) / tmp_target_in_block;
            shared_size = neighbor_group_size * 2 * tmp_target_in_block * sizeof(int);
            // assert(d_ptr_scheduled != NULL);
            // checkCudaErrors(cudaMemset(vout, 0, num_v * feat_out * sizeof(float)));
            // checkCudaErrors(cudaMalloc2((void**)&newval, num_e * sizeof(int)));
        }

        if (scheduled)
        {
            aggr_gat_fine<<<grid, block, shared_size>>>(d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, /*newval*/ d_newval, vin, vout, scalar, vatt,  num_target, feat_in, neighbor_group_size, 0.2f);
            scaleArray <<< (num_v * feat_in + 256 - 1) / 256, 256 >>>
                (vout, scalar, feat_in, num_v);
            // testGPUBuffer(0, scalar);
            // testGPUBuffer(0, d_newval);
        }
        else
        {
            aggr_gat<<<grid, block, shared_size>>>(d_ptr, d_idx, d_val, vin, vout, vatt, num_v, feat_in, 0.2f);
        }
        // checkCudaErrors(cudaDeviceSynchronize());
        // timestamp(t1);
        // dbg(getDuration(t0, t1));
        // return getDuration(t0, t1);
        return 0.0;
    }
    void run_att(float* in_att, float* out_val, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / 32; // 32 threads in a warp
        dim3 attgrid((num_v + tmp_target_in_block - 1) / tmp_target_in_block, 1);
        dim3 attblock(BLOCK_SIZE, 1);
        attGat <<< attgrid, attblock >>> (d_ptr, d_idx, out_val, in_att, num_v, 0.2);
    }
    void run_u_add_v(float* in_att, float* out_val, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / 32; // 32 threads in a warp
        dim3 attgrid((num_v + tmp_target_in_block - 1) / tmp_target_in_block, 1);
        dim3 attblock(BLOCK_SIZE, 1);
        u_add_v<<< attgrid, attblock >>>
            (d_ptr, d_idx, out_val, in_att, num_v);
    }
    void run_add_to_center(float* in_val, float* out_att, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / 32; // 32 threads in a warp
        dim3 attgrid((num_v + tmp_target_in_block - 1) / tmp_target_in_block, 1);
        dim3 attblock(BLOCK_SIZE, 1);
        add_to_center<<< attgrid, attblock >>>
            (d_ptr, d_idx, in_val, out_att, num_v);
    }
    void run_div_each(float* in_att, float* in_out_val, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / 32; // 32 threads in a warp
        dim3 attgrid((num_v + tmp_target_in_block - 1) / tmp_target_in_block, 1);
        dim3 attblock(BLOCK_SIZE, 1);
        each_div<<< attgrid, attblock >>>
            (d_ptr, d_idx, in_out_val, in_att, num_v);
    }
    void run_bwd(float* output, float* doutput, float* newval, float* div, float* infeat, float* d_a_b, float* d_feat, float relu_l, int BLOCK_SIZE)
    {
        int tmp_target_in_block = BLOCK_SIZE / 32; // 32 threads in a warp
        dim3 attgrid((num_target + tmp_target_in_block - 1) / tmp_target_in_block, 1);
        dim3 attblock(BLOCK_SIZE, 1);
        int shared_size = 3 * neighbor_group_size * tmp_target_in_block * sizeof(float);
        aggr_gat_fine_bwd <<< attgrid, attblock, shared_size >>> 
        (d_ptr_scheduled, d_idx_scheduled, d_target_scheduled, output, doutput, newval, div, infeat, d_a_b, d_feat, relu_l, num_target, feat_in, neighbor_group_size);
    }
// __global__ void aggr_gat_fine_bwd(int* ptr, int* idx, int* targetv, float* output, float* doutput, float* newval/*exp(relu(a+b))*/, /*float* addval // before relu*/ float* div, float* infeat, /*float* a_b (a+b) on_edge*/ /*outputs*/ float* d_a_b, float* d_feat, float relu_l, int num_v, int INFEATURE, int NEIGHBOR_NUM)

private:
    float *d_val = NULL;
    float* scalar = NULL;
    float* d_newval = NULL;
};


#endif
