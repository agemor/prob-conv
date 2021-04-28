//
// Created by hyunjk on 4/28/21.
//

#ifndef FRESCO_PREFIXSUM_CUH
#define FRESCO_PREFIXSUM_CUH


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>


void sum_scan_naive(unsigned int *const d_out,
                    const unsigned int *const d_in,
                    const size_t numElems);

void sum_scan_blelloch(unsigned int *const d_out,
                       const unsigned int *const d_in,
                       const size_t numElems);

#endif //FRESCO_PREFIXSUM_CUH
