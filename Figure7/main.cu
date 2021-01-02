#include "util.h"
#include "data.h"
// #include <tensor.h>

// #include "config.h"
// #include "aggr_kernel_no_template.h"
// #include "spmm.h"
// #include "att_kernel_no_template.h"
// #include "dense.h"
// #include "aggr_dense_kernel.h"
#include <queue>

__global__ void makex(int* neis, int feature_len, int mmax, float* input, float* output)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if(tid < mmax)
    {
        int whichnei = neis[tid / feature_len];
        output[tid] = input[whichnei * feature_len + tid % feature_len];
    }
}

__global__ void makex2(int* neis, int feature_len, int mmax, int num_v, int nei_num, float* input, float* output)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int twhich = tid / feature_len;
    int tlane = tid % feature_len;
    if(tid < mmax)
    {
        int iterid = twhich % nei_num;
        int target_pos = (iterid * num_v * feature_len) + (twhich / nei_num * feature_len) + tlane;
        int src_pos = neis[twhich] * feature_len + tlane;
        output[target_pos] = input[src_pos];
    }
}

__forceinline__ __device__ float device_logistic ( float x ) {
	return __frcp_rn ( ( float ) 1 + __expf ( -x ) );
}

__forceinline__ __device__ float device_tanh ( float x ) {
	return tanhf ( x );
}

__global__ void kernel_elementwise_lstm_forward (
	float *__restrict__ g,
	float *__restrict__ g2,
	float *__restrict__ h,
	float *__restrict__ c,
	float *__restrict__ ct,
	float *__restrict__ prev_c,
	int *__restrict__ n_idx,
	int num_v, int feature, int num_nei, int cur_nei) {
    const int i_gates = 3 * num_v * feature;
    const int o_gates = 0;
    const int f_gates = 2 * num_v * feature;
    const int c_gates = num_v * feature;
	size_t elements = num_v * feature;
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	/* in - gates after SGEMMs */
	if ( tid < elements ) {
        int which = n_idx[(tid / feature) * num_nei + cur_nei];
        int idx = which * feature + tid % feature;
        float tmp0 = g2[i_gates + tid] + g[i_gates + idx];
        float tmp1 = g2[o_gates + tid] + g[o_gates + idx];
        float tmp2 = g2[f_gates + tid] + g[f_gates + idx];
        float tmp3 = g2[c_gates + tid] + g[c_gates + idx];
        tmp0 = device_logistic(tmp0);
        tmp1 = device_logistic(tmp1);
        tmp2 = device_logistic(tmp2);
        tmp3 = device_tanh(tmp3);
        float tmp5 = tmp2 * prev_c[tid] + tmp0 * tmp3;
        float tmp6 = device_tanh(tmp5);
        h[tid] = tmp1 * tmp6;
        ct[tid] = tmp6;
        c[tid] = tmp5;
		g2[i_gates + tid] 	= tmp0;
		g2[o_gates + tid] 	= tmp1;
		g2[f_gates + tid] 	= tmp2;
		g2[c_gates + tid] 	= tmp3;
	}
	/* out - updated c and h */
}

__global__ void kernel_elementwise_lstm_forward_dense (
	float *__restrict__ g,
	float *__restrict__ g2,
	float *__restrict__ h,
	float *__restrict__ c,
	float *__restrict__ ct,
	float *__restrict__ prev_c,
	int num_v, int feature, int nei_num, int cur_nei) {
    const int i_gates = 3 * num_v * feature;
    const int o_gates = 0;
    const int f_gates = 2 * num_v * feature;
    const int c_gates = num_v * feature;
	size_t elements = num_v * feature;
	/* there are N * B threads */
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	/* in - gates after SGEMMs */
	if ( tid < elements ) {
        float tmp0 = g2[i_gates + tid] + g[i_gates + tid];
        float tmp1 = g2[o_gates + tid] + g[o_gates + tid];
        float tmp2 = g2[f_gates + tid] + g[f_gates + tid];
        float tmp3 = g2[c_gates + tid] + g[c_gates + tid];
        tmp0 = device_logistic(tmp0);
        tmp1 = device_logistic(tmp1);
        tmp2 = device_logistic(tmp2);
        tmp3 = device_tanh(tmp3);
        float tmp5 = tmp2 * prev_c[tid] + tmp0 * tmp3;
        float tmp6 = device_tanh(tmp5);
        h[tid] = tmp1 * tmp6;
        ct[tid] = tmp6;
        c[tid] = tmp5;
		g2[i_gates + tid] 	= tmp0;
		g2[o_gates + tid] 	= tmp1;
		g2[f_gates + tid] 	= tmp2;
		g2[c_gates + tid] 	= tmp3;
	}
	/* out - updated c and h */
}

enum PTRS
{
    x,
    x_transformed,
    h,
    c,
    weight,
    weight2,
    g2,
    ct,
    x_reorder
};


int main(int argc, char ** argv)
{
    argParse(argc, argv);
    const int times = 5;

    const int BLOCK_SIZE = 512;
    int NEIGHBOR_NUM = 16; // selected neighbor number
    if(NEINUM != -1) NEIGHBOR_NUM = NEINUM;
    curandGenerator_t curand;
    curandCreateGenerator(&curand, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand, 123ULL);
    
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);

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
    vector<int> sizes = 
    {
        n * feature_len,
        n * feature_len * 4,
        n * feature_len * (NEIGHBOR_NUM + 1),
        n * feature_len * (NEIGHBOR_NUM + 1),
        feature_len * feature_len * 4,
        feature_len * feature_len * 4,
        feature_len * n * 4,
        n * feature_len,
        //n * feature_len * (NEIGHBOR_NUM + 1) * 4,
        n * feature_len * (NEIGHBOR_NUM)
    };
    double overall_size = 0;
    for(auto item : sizes)
    {
        float* tmp = NULL;
        checkCudaErrors(cudaMalloc2((void**)&tmp, sizeof(float) * (size_t)item));
        checkCudaErrors(cudaDeviceSynchronize());
        curandGenerateNormal(curand, tmp, item, 0.f, 1.00);
        ptr.push_back(tmp);
        checkCudaErrors(cudaDeviceSynchronize());
        overall_size += item;
    }
    dbg(overall_size);
    const float alpha = 1.0f;
    const float beta = 0.0f;

    {
    checkCudaErrors(cudaDeviceSynchronize());
    double timing_our = 0;
    for(int i = 0; i < times; ++i)
    {
        timestamp(t0);
        checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
            n,  feature_len * 4, feature_len, &alpha,
            ptr[x], n,
            ptr[weight], feature_len,
            &beta, 
            ptr[x_transformed], n));
        for(int iter = 0; iter < NEIGHBOR_NUM; ++iter)
        {
            //dbg(iter);
            checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, feature_len * 4, feature_len, &alpha,
                ptr[h] + iter * n * feature_len, n,
                ptr[weight2], feature_len,
                &beta, 
                ptr[g2], n));
                //ptr[g2] + (iter + 1) * n * feature_len * 4, n));
            //checkCudaErrors(cudaDeviceSynchronize());
            kernel_elementwise_lstm_forward <<< (n * feature_len + BLOCK_SIZE - 1)/ BLOCK_SIZE, BLOCK_SIZE >>>
            (
                ptr[x_transformed], 
                ptr[g2], 
                ptr[h] + (iter + 1) * n * feature_len,
                ptr[c] + (iter + 1) * n * feature_len,
                ptr[ct],
                //ptr[ct] + (iter + 1) * n * feature_len,
                ptr[c] + (iter) * n * feature_len,
                gidxs[0],
                n,
                feature_len,
                NEIGHBOR_NUM,
                iter
            );
            //checkCudaErrors(cudaDeviceSynchronize());
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        if(i > 2) timing_our += getDuration(t0, t1);
    }
    dbg(timing_our / (times - 3));
    }

    {
    checkCudaErrors(cudaDeviceSynchronize());
    double timing_dgl = 0;
    for(int i = 0; i < times; ++i)
    {
        timestamp(t0);
        makex2 <<< (n * NEIGHBOR_NUM * feature_len + 255) / 256, 256 >>>
            (gidxs[0], feature_len, n * NEIGHBOR_NUM * feature_len, n, NEIGHBOR_NUM, ptr[x], ptr[x_reorder]);
        for(int iter = 0; iter < NEIGHBOR_NUM; ++iter)
        {
            checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, feature_len * 4, feature_len, &alpha,
                // ptr[h], n,
                ptr[h] + iter * n * feature_len, n,
                ptr[weight2], feature_len,
                &beta, 
                ptr[g2], n));
                //ptr[g2] + (iter + 1) * n * feature_len * 4, n));
            checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n,  feature_len * 4, feature_len, &alpha,
                ptr[x_reorder] + feature_len * n * iter, n,
                ptr[weight], feature_len,
                &beta, 
                ptr[x_transformed], n));
            kernel_elementwise_lstm_forward_dense <<< (n * feature_len + BLOCK_SIZE - 1)/ BLOCK_SIZE, BLOCK_SIZE >>>
            (
                ptr[x_transformed], 
                ptr[g2], 
                //ptr[g2] + (iter + 1) * n * feature_len * 4, 
                ptr[h] + (iter + 1) * n * feature_len,
                ptr[c] + (iter + 1) * n * feature_len,
                ptr[ct],
                //ptr[ct] + (iter + 1) * n * feature_len,
                ptr[c] + (iter) * n * feature_len,
                n,
                feature_len,
                NEIGHBOR_NUM,
                iter
            );
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        if(i > 2) timing_dgl += getDuration(t0, t1);
    }
    dbg(timing_dgl / (times - 3));
    }

    {
    checkCudaErrors(cudaDeviceSynchronize());
    double timing_sparsefetch = 0;
    for(int i = 0; i < times; ++i)
    {
        timestamp(t0);
        for(int iter = 0; iter < NEIGHBOR_NUM; ++iter)
        {
            checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n, feature_len * 4, feature_len, &alpha,
                // ptr[h], n,
                ptr[h] + iter * n * feature_len, n,
                ptr[weight2], feature_len,
                &beta, 
                ptr[g2], n));
                // ptr[g2] + (iter + 1) * n * feature_len * 4, n));
            checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
                n,  feature_len * 4, feature_len, &alpha,
                ptr[x_reorder] + feature_len * n * iter, n,
                ptr[weight], feature_len,
                &beta, 
                ptr[x_transformed], n));
            kernel_elementwise_lstm_forward_dense <<< (n * feature_len + BLOCK_SIZE - 1)/ BLOCK_SIZE, BLOCK_SIZE >>>
            (
                ptr[x_transformed], 
                ptr[g2],
                //ptr[g2] + (iter + 1) * n * feature_len * 4, 
                ptr[h] + (iter + 1) * n * feature_len,
                ptr[c] + (iter + 1) * n * feature_len,
                ptr[ct],
                //ptr[ct] + (iter + 1) * n * feature_len,
                ptr[c] + (iter) * n * feature_len,
                n,
                feature_len,
                NEIGHBOR_NUM,
                iter
            );
        }
        checkCudaErrors(cudaDeviceSynchronize());
        timestamp(t1);
        if(i > 2) timing_sparsefetch += getDuration(t0, t1);
    }
    dbg(timing_sparsefetch / (times - 3));
    }
}
