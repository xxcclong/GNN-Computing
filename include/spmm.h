#ifndef SPMM_H
#define SPMM_H

// #define LENFEATURE 64
#define TB 128

// #define LENFEATURE2 64
// #define MAXNUMNEIGHBOR 294
// #define TB2 32

__global__ void validate2(float *ref, float *ans, int num, int *diffnum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num)
    {
        if (abs((ref[tid] - ans[tid]) / ref[tid]) > 1e-2)
        {
            atomicAdd(diffnum, 1);
        }
    }
}

__global__ void validateReordered(float *ref, float *ans, int *map, int num_v, int feature_len, int *diffnum)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_v * feature_len)
    {
        if (abs((ref[tid] - ans[map[(tid / feature_len)] * feature_len + tid % feature_len])) > 1e-2)
        {
            atomicAdd(diffnum, 1);
        }
    }
}

int valid(float *y, float *y2, int num)
{
    int *diffnum;
    checkCudaErrors(cudaMalloc2((void **)&diffnum, 1 * sizeof(int)));
    checkCudaErrors(cudaMemset(diffnum, 0, sizeof(int)));
    checkCudaErrors(cudaDeviceSynchronize());
    // validate<<<(feature_len * n + 127) / 128, 128 >>> (y, y2, feature_len * n);
    validate2<<<(num + 127) / 128, 128>>>(y, y2, num, diffnum);
    checkCudaErrors(cudaDeviceSynchronize());
    int ans = 33;
    checkCudaErrors(cudaMemcpy(&ans, diffnum, 1 * sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    // if(ans > num / 100 + 5)
    // {
    //     testGPUBuffer(0, y, 100);
    //     testGPUBuffer(0, y2, 100);

    //     float* a = new float[num];
    //     float* b = new float[num];
    //     checkCudaErrors(cudaMemcpy(a, y, num * sizeof(float), cudaMemcpyDeviceToHost));
    //     checkCudaErrors(cudaMemcpy(b, y2, num * sizeof(float), cudaMemcpyDeviceToHost));
    //     int cnt = 0;
    //     for(int i = 0; i < num; ++i)
    //     {
    //         if(abs( (a[i] - b[i]) / a[i]) > 1e-2)
    //         {
    //             printf("diff=%d %f %f\n", i, a[i], b[i]);
    //             ++cnt;
    //             if(cnt > 10)
    //             break;
    //         }
    //     }
    // }
    return ans;
}

int validReordered(float *y, float *y2, int num_v, int feature_len)
{
    if (rows == NULL)
        return valid(y, y2, num_v * feature_len);
    else
    {
        int *d_map = NULL;
        checkCudaErrors(cudaMalloc2((void **)&d_map, num_v * sizeof(int)));
        checkCudaErrors(cudaMemcpy(d_map, rows, num_v * sizeof(int), cudaMemcpyHostToDevice));
        int *diffnum;
        checkCudaErrors(cudaMalloc2((void **)&diffnum, 1 * sizeof(int)));
        checkCudaErrors(cudaMemset(diffnum, 0, sizeof(int)));
        checkCudaErrors(cudaDeviceSynchronize());
        validateReordered<<<(num_v * feature_len + 127) / 128, 128>>>(y, y2, d_map, num_v, feature_len, diffnum);
        checkCudaErrors(cudaDeviceSynchronize());
        int ans = 33;
        checkCudaErrors(cudaMemcpy(&ans, diffnum, 1 * sizeof(int), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaDeviceSynchronize());
        return ans;
    }
}

// __global__
// void spmm2(int numV, int* ptr, int* idx, float* val, float* denseInput, float* denseOutput)
// {
//     __shared__ int whichs[MAXNUMNEIGHBOR];
//     __shared__ float values[MAXNUMNEIGHBOR];
//     const int bid = blockIdx.x;
//     const int tidinb = threadIdx.x;
//     int begin = ptr[bid];
//     int end = ptr[bid + 1];
//     int num_neighbor = end - begin;
//     // if(num_neighbor > MAXNUMNEIGHBOR)
//     // {
//     //     printf("num neighbor %d\n", num_neighbor);
//     //     return;
//     // }
//     const int step0 = (num_neighbor + TB2 - 1) / TB2;
//     const int inbegin0 = step0 * tidinb;
//     int inend0 = inbegin0 + step0;
//     if(inend0 > num_neighbor) inend0 = num_neighbor;
//     #pragma unroll
//     for(int i = inbegin0; i < inend0; ++i)
//     {
//         whichs[i] = idx[i + begin];
//         values[i] = val[i + begin];
//     }

//     const int step = (LENFEATURE2 + TB2 - 1) / TB2;
//     int inbegin = step * tidinb;
//     int inend = inbegin + step;
//     if(inend > LENFEATURE2)
//     {
//         inend = LENFEATURE2;
//     }
//     float ans[step];
//     __syncthreads();
//     int base = whichs[0] * LENFEATURE2;
//     float theval = values[0];
//     #pragma unroll
//     for(int i = inbegin; i < inend; ++i)
//     {
//         ans[i] = denseInput[base + i] * theval;
//         // ans[i - inbegin] = denseInput[base + i] * theval;
//     }
//     #pragma unroll
//     for(int nei = 1; nei < num_neighbor; ++nei)
//     {
//         base = whichs[nei] * LENFEATURE2;
//         theval = values[nei];
//         #pragma unroll
//         for(int i = inbegin; i < inend; ++i)
//         {
//             ans[i] += denseInput[base + i] * theval;
//         }
//     }
//     base = bid * LENFEATURE2;
//     #pragma unroll
//     for(int i = inbegin; i < inend; ++i)
//     {
//         denseOutput[base + i] = ans[i];
//     }
// }

// __global__
// void pull_spmm_whole(int numV, int* ptr, int* idx, float* val, float* denseInput, float* denseOutput, int beginv, int endv, bool firstKernel = false)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     float ans[LENFEATURE];
//     if(tid < numV)
//     {
//         const int begin = ptr[tid], end = ptr[tid + 1];
//         // if(tid == 0) printf("%d %d\n", begin, end);
//         int begin2 = -1;
//         for(begin2 = begin; begin2 < end; ++begin2)
//         {
//             int whichNeighbor = idx[begin2];
//             if(whichNeighbor < beginv || whichNeighbor >= endv)
//             {
//                 continue;
//             }
//             else
//             {
//                 const float value = val[begin2];
//                 const int base = LENFEATURE * whichNeighbor;
//                 #pragma unroll
//                 for(int j = 0; j < LENFEATURE; ++j)
//                 {
//                     ans[j] = value * denseInput[base + j];
//                 }
//                 break;
//             }
//         }
//         for(int i = begin2 + 1; i < end; ++i)
//         {
//             int whichNeighbor2 = idx[i];
//             if(whichNeighbor2 < beginv || whichNeighbor2 >= endv)
//             {
//                 // printf("error! %d %d %d\n", whichNeighbor2, beginv, endv);
//             //     // while(whichNeighbor2 != 0)
//             //     // {
//             //     //     whichNeighbor2 -= 1;
//             //     // }
//                 continue;
//             }
//             const float value = val[i];
//             const int base = LENFEATURE * whichNeighbor2;
//             #pragma unroll
//             for(int j = 0; j < LENFEATURE; ++j)
//             {
//                 ans[j] += value * denseInput[base + j];
//             }
//         }
//         if(firstKernel)
//         {
//             #pragma unroll
//             for(int j = 0; j < LENFEATURE; ++j)
//             {
//                 denseOutput[LENFEATURE * tid + j] = ans[j];
//             }
//         }
//         else
//         {
//             #pragma unroll
//             for(int j = 0; j < LENFEATURE; ++j)
//             {
//                 denseOutput[LENFEATURE * tid + j] += ans[j];
//             }
//         }
//     }
// }

template <int LENFEATURE>
__global__ void spmm(int numV, int *ptr, int *idx, float *val, float *denseInput, float *denseOutput)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ans[LENFEATURE];
    // #pragma unroll
    // for(int j = 0; j < LENFEATURE; ++j)
    //     {
    //         ans[j] = 0;
    //     }
    if (tid < numV)
    {
        const int begin = ptr[tid], end = ptr[tid + 1];
        if (begin == end)
            return;
        // if(tid == 0) printf("%d %d\n", begin, end);
        const int whichNeighbor = idx[begin];
        const float value = val[begin];
        const int base = LENFEATURE * whichNeighbor;
#pragma unroll
        for (int j = 0; j < LENFEATURE; ++j)
        {
            ans[j] = value * denseInput[base + j];
        }

        for (int i = begin + 1; i < end; ++i)
        {
            const int whichNeighbor = idx[i];
            const float value = val[i];
            const int base = LENFEATURE * whichNeighbor;
#pragma unroll
            for (int j = 0; j < LENFEATURE; ++j)
            {
                ans[j] += value * denseInput[base + j];
            }
        }
#pragma unroll
        for (int j = 0; j < LENFEATURE; ++j)
        {
            denseOutput[LENFEATURE * tid + j] = ans[j];
        }
    }
}

// __global__
// void spmm_feature(int numV, int* ptr, int* idx, float* val, float* denseInput, float* denseOutput, int feature_len)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     float ans[LENFEATURE];
//     if(tid < numV)
//     {
//         const int begin = ptr[tid], end = ptr[tid + 1];
//         // if(tid == 0) printf("%d %d\n", begin, end);
//         const int whichNeighbor = idx[begin];
//         const float value = val[begin];
//         const int base = feature_len * whichNeighbor;
//         #pragma unroll
//         for(int j = 0; j < feature_len; ++j)
//         {
//             ans[j] = value * denseInput[base + j];
//         }
//         for(int i = begin + 1; i < end; ++i)
//         {
//             const int whichNeighbor = idx[i];
//             const float value = val[i];
//             const int base = feature_len * whichNeighbor;
//             #pragma unroll
//             for(int j = 0; j < feature_len; ++j)
//             {
//                 ans[j] += value * denseInput[base + j];
//             }
//         }
//         #pragma unroll
//         for(int j = 0; j < feature_len; ++j)
//         {
//             denseOutput[feature_len * tid + j] = ans[j];
//         }
//     }
// }

#endif