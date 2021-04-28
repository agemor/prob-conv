//
// Created by hyunjk on 4/27/21.
//

#include "Utils.cuh"
#include <iostream>

namespace fresco {

    void gemm(const float *mat_a, const float *mat_b, float *mat_out, const int m, const int k, const int n) {
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;

        // Create mat_a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        //cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, mat_a, m, mat_b, k, beta, mat_out, m);
        // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, alpha, mat_b, n, mat_a, k, beta, mat_out, n);

        // Destroy the handle
        cublasDestroy(handle);
    }


}
