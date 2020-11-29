#ifndef DENSE_H
#define DENSE_H
#include "util.h"
void matmul_NN(float* A, float* B, float* C, int inM, int inN, int inK, float* tmp)
{
    assert(cublasHs[0] != NULL);
    const float alpha = 1;
    const float beta = 0;
    checkCudaErrors(cublasSgemm(cublasHs[0], CUBLAS_OP_T, CUBLAS_OP_T,
        inM, inN, inK,
        &alpha,
        A, inK,
        B, inN,
        &beta,
        tmp, inM));
    checkCudaErrors(cublasSgeam(cublasHs[0], CUBLAS_OP_T, CUBLAS_OP_N,
        inN, inM,
        &alpha,
        tmp, inM,
        NULL,
        NULL, inN,
        C, inN));
}
#endif