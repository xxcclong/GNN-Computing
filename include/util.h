#ifndef UTIL_H
#define UTIL_H

#include <cusparse.h>
#include <cublas_v2.h>
#include <iostream>
#include <cuda_profiler_api.h>
#include <curand.h>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <vector>
#include <chrono>
#include <assert.h>
#include <sstream>
#include <assert.h>
#include <pthread.h>
#include <omp.h>
#include <string>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <random>

// #include "nccl.h"
#include "dbg.h"
#include "args.hxx"

using namespace std;

// #define DENSEADJ
// #define cuda10
// #define ERROR_PROF
// #define CONTINUE_DATA
// #define GRAPH_ANALYSIS

#define CEIL(a, b) (((a) + (b)-1) / (b))

extern int GPUNUM;
extern int NEINUM;

// extern ncclComm_t* comms;
extern int n, m, feature_len;
extern int *rows;
extern int *reverse_rows;
extern vector<void *> registered_ptr;
extern int outfea;
extern int total_size;

extern string inputfeature;
extern string inputweight;
extern string inputgraph;
extern string edgefile;
extern string ptrfile;
extern string partitionfile;
extern string reorderfile;
extern string inputtransgraph;
extern string partialgraphs;


// ************************************************************
// variables for single train
extern int *gptr, *gidx;
extern float *gval;
extern cublasHandle_t cublasH;
extern cudaStream_t stream;
// ************************************************************
// var for multi train
extern int *numVertex, *numEdge;
extern int **gptrs, **gidxs;
extern cublasHandle_t *cublasHs;
// void ncclInit();
inline bool fexist(const std::string &name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}
void argParse(int argc, char **argv, int *p_limit = NULL, int *p_limit2 = NULL);

#define timestamp(__var__) auto __var__ = std::chrono::system_clock::now();

#define FatalError(s)                                     \
    do                                                    \
    {                                                     \
        std::stringstream _where, _message;               \
        _where << __FILE__ << ':' << __LINE__;            \
        _message << std::string(s) + "\n"                 \
                 << __FILE__ << ':' << __LINE__;          \
        std::cerr << _message.str() << "\nAborting...\n"; \
        cudaDeviceReset();                                \
        exit(1);                                          \
    } while (0)


#define checkCudaErrors(status)                   \
    do                                            \
    {                                             \
        std::stringstream _error;                 \
        if (status != 0)                          \
        {                                         \
            _error << "Cuda failure: " << status; \
            FatalError(_error.str());             \
        }                                         \
    } while (0)

// #define checkCusparseErrors(status) do {                                   \
//     std::stringstream _error;                                          \
//     if (status != 0) {                                                 \
//       _error << "Cusparse failure: " << cusparseGetErrorString(status);                            \
//       FatalError(_error.str());                                        \
//     }                                                                  \
// } while(0)

inline double getDuration(std::chrono::time_point<std::chrono::system_clock> a,
                          std::chrono::time_point<std::chrono::system_clock> b)
{
    return std::chrono::duration<double>(b - a).count();
}

inline double getFLOP(double time)
{
    assert(time > 0);
    assert(m > 0);
    assert(feature_len > 0);
    double ans = m * feature_len * 2 / time / 1e9;
    return ans;
    // return m * feature_len * 2 / time / 1e9;
}

static inline unsigned int roundUp(unsigned int nominator, unsigned int denominator)
{
    return (nominator + denominator - 1) / denominator;
}

static inline void syncAll(int num)
{
    for (int i = 0; i < num; ++i)
    {
        checkCudaErrors(cudaSetDevice(i));
        checkCudaErrors(cudaDeviceSynchronize());
    }
}

inline cudaError_t cudaMalloc2(void **a, size_t s)
{
    if (s == 0)
        return (cudaError_t)(0);
    total_size += s;
    // dbg(total_size);
    return cudaMalloc(a, ((s + 511) / 512) * 512);
    //return cudaMalloc(a, ((s+511)/512)*512);
}

template <class T>
inline void registerPtr(T ptr)
{
    registered_ptr.push_back((void *)(ptr));
}

template <class T>
void safeFree(T *&a)
{
    for (auto item : registered_ptr)
    {
        if (((void *)(a)) == item)
            return;
    }
    if (a != NULL)
    {
        cudaFree(a);
        cudaGetLastError();
        a = NULL;
    }
    else
    {
    }
}

template <class T>
T *createCopy(T *p, int size)
{
    T *p_d;
    checkCudaErrors(cudaMalloc2((void **)&p_d, size * sizeof(T)));
    checkCudaErrors(cudaMemcpy(p_d, p, sizeof(T) * size, cudaMemcpyHostToDevice));
    return p_d;
}

template <class T>
void copyVec2Dev(std::vector<T> *vec, T *&output)
{
    assert(output == NULL);
    checkCudaErrors(cudaMalloc2((void **)&output, vec->size() * sizeof(T)));
    checkCudaErrors(cudaMemcpy(output, vec->data(), vec->size() * sizeof(T), cudaMemcpyHostToDevice));
    vector<T>().swap(*vec); // release the memory of the vec
}

struct CSR
{
    CSR(int *outptr, int *outidx, float *outval) : ptr(outptr), idx(outidx), val(outval) {}
    int *ptr;
    int *idx;
    float *val;
};

class CSRSubGraph
{
public:
    CSRSubGraph(int *outvertexset, int *outptr, int *outidx, int vertex_num, int edge_num) : vertexset(outvertexset), ptr(outptr), idx(outidx), num_v(vertex_num), num_e(edge_num) {}
    void free()
    {
        safeFree(vertexset);
        safeFree(ptr);
        safeFree(idx);
    }

    int *vertexset = NULL;
    int *ptr = NULL;
    int *idx = NULL;
    int num_v = 0;
    int num_e = 0;
};

#endif
