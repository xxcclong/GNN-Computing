#ifndef DATA_H
#define DATA_H
#include "util.h"
// #include "partition.h"

float *createCudaMatrixRandom(int n, const char *name = NULL);

template <class T>
T *createCudaMatrixCopy(T *d, int n)
{
    T *p_d;
    // cudaMallocManaged(&p_d, n * sizeof(T));
    cudaMalloc2((void **)&p_d, n * sizeof(T));
    cudaMemcpy(p_d, d, n * sizeof(T), cudaMemcpyHostToDevice);
    return p_d;
}

template <class T>
void convert(T *a, int l1, int l2);

template <class T>
void testGPUBuffer(int gpuid, T *dptr, int outnum = 64)
{
    checkCudaErrors(cudaSetDevice(gpuid));
    checkCudaErrors(cudaDeviceSynchronize());
    const int nums = outnum;
    T temp[nums];
    checkCudaErrors(cudaMemcpy(temp, dptr, nums * sizeof(T), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
    printf("gpu %d\n", gpuid);
    for (int j = 0; j < nums; ++j)
    {
        if (j % 32 == 0)
            cout << endl;
        cout << temp[j] << ' ';
    }
    cout << '\n';
}

float *createLabelCPU();

float *createLabel();

void reorderCSR(int *ptr, int *index, int *map, int *reverse_map);

void load_graph(std::string dset, int &num_v, int &num_e, int *&indptr, int *&indices, bool shuffle = true, std::string reoreder_subfix = "");

void prepareDataMulti();

void prepareGraphMultiFast();

void prepareGraphMulti();

void prepareGraphPartial();

void prepareGraphPartial4All();

void prepareData();
#endif
