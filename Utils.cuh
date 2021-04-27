//
// Created by hyunjk on 4/27/21.
//

#ifndef FRESCO_UTILS_CUH
#define FRESCO_UTILS_CUH

#include <cublas_v2.h>

namespace fresco {

    const int NUM_THREADS = 1024;

    inline int getNumBlock(int n, int threads) {
        return (n - 1) / threads + 1;
    }


    // mat_out(m,n) = mat_a(m,k) * mat_b(k,n)
    void gemm(const float *mat_a, const float *mat_b, float *mat_out, const int m, const int k, const int n);

}

#endif //FRESCO_UTILS_CUH
