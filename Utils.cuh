//
// Created by hyunjk on 4/27/21.
//

#ifndef FRESCO_UTILS_CUH
#define FRESCO_UTILS_CUH

#include <cublas_v2.h>
#include <iostream>

namespace fresco {

    const int NUM_THREADS = 1024;

    inline int getNumBlock(int n, int threads) {
        return (n - 1) / threads + 1;
    }


    // mat_out(m,n) = mat_a(m,k) * mat_b(k,n)
    void gemm(const float *mat_a, const float *mat_b, float *mat_out, const int m, const int k, const int n);

    template<typename T>
    void printData(const T *d_data, uint num_elem, uint print_limit = 0) {
        auto *h_data = (T *) malloc(num_elem * sizeof(T));
        cudaMemcpy(h_data, d_data, num_elem * sizeof(T), cudaMemcpyDeviceToHost);

        if (print_limit == 0) {
            print_limit = num_elem;
        }

        for (int i = 0; i < print_limit; ++i) {
            std::cout << h_data[i] << ' ';
        }
        std::cout << std::endl;
    }


}

#endif //FRESCO_UTILS_CUH
