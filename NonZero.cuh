//
// Created by hyunjk on 4/27/21.
//

#ifndef FRESCO_NONZERO_CUH
#define FRESCO_NONZERO_CUH

#include "Utils.cuh"
#include <thrust/scan.h>
#include <thrust/device_vector.h>
#include "PrefixSum.cuh"

int get_nonzero_indices(unsigned int *d_input, unsigned int *d_output, int length, int blockSize);

#endif //FRESCO_NONZERO_CUH
