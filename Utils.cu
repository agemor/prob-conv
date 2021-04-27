//
// Created by hyunjk on 4/27/21.
//

#include "Utils.cuh"


namespace fresco {

    void gemm(const float *mat_a, const float *mat_b, float *mat_out, const int m, const int k, const int n) {
        int lda = m, ldb = k, ldc = m;
        const float alf = 1;
        const float bet = 0;
        const float *alpha = &alf;
        const float *beta = &bet;



        // Create mat_a handle for CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Do the actual multiplication
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, mat_a, lda, mat_b, ldb, beta, mat_out, ldc);

        // Destroy the handle
        cublasDestroy(handle);
    }
}
