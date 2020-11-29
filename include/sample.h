#ifndef SAMPLE_H
#define SAMPLE_H
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "util.h"
#include <curand.h>
#include <curand_kernel.h>

__global__ void fix_prefix_sum(int *arr, int *output, int num)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid == 0)
        output[num] = output[num - 1] + arr[num - 1];
}

__global__ void compact(int *prefix_sum, int *active, int *output, int num_v)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_v && active[tid])
    {
        output[prefix_sum[tid]] = tid;
    }
}

__global__ void compact_withnum(int *prefix_sum, int *active, int *arr_to_compact, int *output, int num_v)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_v && active[tid])
    {
        output[prefix_sum[tid] + 1] = arr_to_compact[tid + 1];
        if (prefix_sum[tid] == 0)
        {
            output[0] = 0;
        }
    }
}

__global__ void getSubDegree(int *ptr, int *active, int *output, int num_v)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_v && active[tid])
    {
        output[tid] = ptr[tid + 1] - ptr[tid];
    }
}

__global__ void getSubDegreeWithSample(int *ptr, int *active, int *output, int num_v, int limit)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < num_v && active[tid])
    {
        int num = ptr[tid + 1] - ptr[tid];
        if (num > limit)
            num = limit;
        output[tid] = num;
    }
}

__global__ void moveEdge(int *selected_vertex, int *new_ptr, int *ptr, int *idx, int *new_idx, int active_num)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int which = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (which < active_num)
    {
        int vid = selected_vertex[which];
        int oldbegin = ptr[vid], oldend = ptr[vid + 1];
        int newbegin = new_ptr[which], newend = new_ptr[which + 1];
        int num = oldend - oldbegin;
        for (int i = lane; i < num; i += 32)
        {
            new_idx[newbegin + i] = idx[oldbegin + i];
        }
    }
}

__global__ void moveEdgeSelective(int *selected_vertex, int *new_ptr, int *ptr, int *idx, int *new_idx, int *chosen, int active_num, int limit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < active_num)
    {
        int vid = selected_vertex[tid];
        int oldbegin = ptr[vid], oldend = ptr[vid + 1];
        int newbegin = new_ptr[tid], newend = new_ptr[tid + 1];
        int num = oldend - oldbegin;
        if (num <= limit)
        {
            for (int i = 0; i < num; ++i)
            {
                new_idx[newbegin + i] = idx[oldbegin + i];
            }
        }
        else
        {
            int cnt = 0;
            const int mark = (int)(num < 2 * limit);
            for (int i = 0; i < num; ++i)
            {
                if (chosen[oldbegin + i] == mark)
                {
                    new_idx[newbegin + cnt] = idx[oldbegin + i];
                    ++cnt;
                }
            }
        }
    }
}

__global__ void expandActive(int *active, int *outactive, int *ptr, int *idx, int num_v)
{
    int warpid = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int which = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    if (which < num_v && active[which])
    {
        if (lane == 0)
            outactive[which] = 1;
        int begin = ptr[which], end = ptr[which + 1];
        for (int i = lane + begin; i < end; i += 32)
        {
            outactive[idx[i]] = 1;
        }
    }
}

CSRSubGraph fullGraph(int *ptr, int *idx)
{
    return CSRSubGraph(NULL, ptr, idx, n, m);
}

CSRSubGraph sampleVertex(int *&active_vertex, int *ptr, int *idx, int layer_num = 1) // all ptrs are on device
{
    const int BLOCK_SIZE = 64;
    const int tmp_target_in_block = BLOCK_SIZE / 32;

    int *outactive = NULL;
    checkCudaErrors(cudaMalloc2((void **)&outactive, n * sizeof(int)));
    for (int i = 0; i < layer_num - 1; ++i)
    {
        checkCudaErrors(cudaMemset(outactive, 0, n * sizeof(int)));
        expandActive<<<(n + tmp_target_in_block - 1) / tmp_target_in_block, BLOCK_SIZE>>>(active_vertex, outactive, ptr, idx, n);
        checkCudaErrors(cudaMemcpy(active_vertex, outactive, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<int> active_vertex_thr(active_vertex);
    int active_num = thrust::reduce(active_vertex_thr, active_vertex_thr + n);
    dbg(active_num);
    dbg(((float)active_num) / n);

    int *active_vertex_prefixsum = NULL;
    checkCudaErrors(cudaMalloc2((void **)&active_vertex_prefixsum, (n + 1) * sizeof(int)));
    int *compacted_vertex = NULL;
    checkCudaErrors(cudaMalloc2((void **)&compacted_vertex, active_num * sizeof(int)));
    int *compacted_degree = NULL;
    checkCudaErrors(cudaMalloc2((void **)&compacted_degree, (active_num + 1) * sizeof(int)));
    int *sub_degree = NULL;
    checkCudaErrors(cudaMalloc2((void **)&sub_degree, n * sizeof(int)));
    int *sub_degree_prefixsum = NULL;
    checkCudaErrors(cudaMalloc2((void **)&sub_degree_prefixsum, (n + 1) * sizeof(int)));

    // step 1
    thrust::device_ptr<int> active_vertex_prefixsum_thr(active_vertex_prefixsum);
    thrust::exclusive_scan(active_vertex_thr, active_vertex_thr + n, active_vertex_prefixsum_thr);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 2
    compact<<<(n + 64 - 1) / 64, 64>>>(active_vertex_prefixsum, active_vertex, compacted_vertex, n);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 3
    getSubDegree<<<(n + 64 - 1) / 64, 64>>>(ptr, active_vertex, sub_degree, n);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 4
    thrust::device_ptr<int> sub_degree_thr(sub_degree);
    thrust::device_ptr<int> sub_degree_prefixsum_thr(sub_degree_prefixsum);
    thrust::exclusive_scan(sub_degree_thr, sub_degree_thr + n, sub_degree_prefixsum_thr);
    fix_prefix_sum<<<1, 1>>>(sub_degree, sub_degree_prefixsum, n);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 5
    compact_withnum<<<(n + 64 - 1) / 64, 64>>>(active_vertex_prefixsum, active_vertex, sub_degree_prefixsum, compacted_degree, n);
    checkCudaErrors(cudaDeviceSynchronize());

    int new_num_edge = 0;
    checkCudaErrors(cudaMemcpy(&new_num_edge, compacted_degree + active_num, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    dbg(new_num_edge);
    dbg(((float)new_num_edge) / m);
    // step 6
    int *new_idx = NULL;
    checkCudaErrors(cudaMalloc2((void **)&new_idx, new_num_edge * sizeof(int)));
    // compacted_v(selected vertex), ptr, real_idx, real_ptr
    moveEdge<<<(active_num + tmp_target_in_block - 1) / tmp_target_in_block, BLOCK_SIZE>>>(compacted_vertex, compacted_degree, ptr, idx, new_idx, active_num);
    // testGPUBuffer(0, new_idx + new_num_edge - 100);
    safeFree(sub_degree_prefixsum);
    safeFree(sub_degree);
    safeFree(active_vertex_prefixsum);
    return CSRSubGraph(compacted_vertex, compacted_degree, new_idx, active_num, new_num_edge);
}

__global__ void initRNG(curandState *const rngStates,
                        const unsigned int seed)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &rngStates[tid]);
}

// thread independent
__global__ void expandActiveRandom(int *active, int *outactive, int *ptr, int *idx, curandState *const rngStates, int *chosen, int *expanded, int num_v, int limit)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_v && active[tid])
    {
        outactive[tid] = 1;
        if (expanded[tid])
            return;
        expanded[tid] = 1;
        curandState local_state = rngStates[tid];
        int begin = ptr[tid], end = ptr[tid + 1];
        int num = end - begin;
        if (num < limit)
        {
            for (int i = begin; i < end; ++i)
                outactive[idx[i]] = 1;
        }
        else
        {
            // return;
            int mark = 1;
            int upperlimit = limit;
            if (num < 2 * limit)
            {
                upperlimit = num - limit;
                mark = 2; // exclude the 2 edges
            }
            int cnt = 0;
            while (cnt != upperlimit)
            {
                int x = curand(&local_state) % num;
                if (!chosen[begin + x])
                {
                    chosen[begin + x] = mark;
                    ++cnt;
                }
            }
            cnt = 0;
            if (mark == 2)
            {
                for (int i = begin; i < end; ++i)
                {
                    if (chosen[i] != mark)
                    {
                        ++cnt;
                        outactive[idx[i]] = 1;
                    }
                }
            }
            else
            {
                for (int i = begin; i < end; ++i)
                {
                    if (chosen[i] == mark)
                    {
                        ++cnt;
                        outactive[idx[i]] = 1;
                    }
                }
            }
        }
    }
}

CSRSubGraph sampleVertexSampleNeighbor(int *&active_vertex, int *ptr, int *idx, int neighbor_num, int layer_num = 1) // all ptrs are on device
{
    const int BLOCK_SIZE = 64;
    const int tmp_target_in_block = BLOCK_SIZE / 32;

    curandState *stat = NULL;
    checkCudaErrors(cudaMalloc2((void **)&stat, n * sizeof(curandState)));
    initRNG<<<(n + BLOCK_SIZE - 1) / 64, BLOCK_SIZE>>>(stat, 123);

    int *chosen = NULL;
    checkCudaErrors(cudaMalloc2((void **)&chosen, m * sizeof(int)));
    checkCudaErrors(cudaMemset(chosen, 0, m * sizeof(int)));

    int *expanded = NULL;
    checkCudaErrors(cudaMalloc2((void **)&expanded, n * sizeof(int)));
    checkCudaErrors(cudaMemset(expanded, 0, n * sizeof(int)));

    int *outactive = NULL;
    checkCudaErrors(cudaMalloc2((void **)&outactive, n * sizeof(int)));
    for (int i = 0; i < layer_num - 1; ++i)
    {
        checkCudaErrors(cudaMemset(outactive, 0, n * sizeof(int)));
        expandActiveRandom<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(active_vertex, outactive, ptr, idx, stat, chosen, expanded, n, neighbor_num);
        checkCudaErrors(cudaMemcpy(active_vertex, outactive, n * sizeof(int), cudaMemcpyDeviceToDevice));
    }
    checkCudaErrors(cudaDeviceSynchronize());

    thrust::device_ptr<int> active_vertex_thr(active_vertex);
    int active_num = thrust::reduce(active_vertex_thr, active_vertex_thr + n);
    dbg(active_num);

    int *active_vertex_prefixsum = NULL;
    checkCudaErrors(cudaMalloc2((void **)&active_vertex_prefixsum, (n + 1) * sizeof(int)));
    int *compacted_vertex = NULL;
    checkCudaErrors(cudaMalloc2((void **)&compacted_vertex, active_num * sizeof(int)));
    int *compacted_degree = NULL;
    checkCudaErrors(cudaMalloc2((void **)&compacted_degree, (active_num + 1) * sizeof(int)));
    int *sub_degree = NULL;
    checkCudaErrors(cudaMalloc2((void **)&sub_degree, n * sizeof(int)));
    int *sub_degree_prefixsum = NULL;
    checkCudaErrors(cudaMalloc2((void **)&sub_degree_prefixsum, (n + 1) * sizeof(int)));

    // step 1
    thrust::device_ptr<int> active_vertex_prefixsum_thr(active_vertex_prefixsum);
    thrust::exclusive_scan(active_vertex_thr, active_vertex_thr + n, active_vertex_prefixsum_thr);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 2
    compact<<<(n + 64 - 1) / 64, 64>>>(active_vertex_prefixsum, active_vertex, compacted_vertex, n);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 3
    getSubDegreeWithSample<<<(n + 64 - 1) / 64, 64>>>(ptr, active_vertex, sub_degree, n, neighbor_num);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 4
    thrust::device_ptr<int> sub_degree_thr(sub_degree);
    thrust::device_ptr<int> sub_degree_prefixsum_thr(sub_degree_prefixsum);
    thrust::exclusive_scan(sub_degree_thr, sub_degree_thr + n, sub_degree_prefixsum_thr);
    fix_prefix_sum<<<1, 1>>>(sub_degree, sub_degree_prefixsum, n);
    // checkCudaErrors(cudaDeviceSynchronize());

    // step 5
    compact_withnum<<<(n + 64 - 1) / 64, 64>>>(active_vertex_prefixsum, active_vertex, sub_degree_prefixsum, compacted_degree, n);
    checkCudaErrors(cudaDeviceSynchronize());

    int new_num_edge = 0;
    checkCudaErrors(cudaMemcpy(&new_num_edge, compacted_degree + active_num, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    dbg(new_num_edge);
    // step 6
    int *new_idx = NULL;
    checkCudaErrors(cudaMalloc2((void **)&new_idx, new_num_edge * sizeof(int)));
    // compacted_v(selected vertex), ptr, real_idx, real_ptr
    moveEdgeSelective<<<(active_num + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(compacted_vertex, compacted_degree, ptr, idx, new_idx, chosen, active_num, neighbor_num);
    checkCudaErrors(cudaDeviceSynchronize());
    // testGPUBuffer(0, new_idx + new_num_edge - 100);
    safeFree(sub_degree_prefixsum);
    safeFree(sub_degree);
    safeFree(active_vertex_prefixsum);
    safeFree(chosen);
    safeFree(outactive);
    safeFree(expanded);
    return CSRSubGraph(compacted_vertex, compacted_degree, new_idx, active_num, new_num_edge);
}
#endif