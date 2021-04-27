//
// Created by hyunjk on 4/27/21.
//

#ifndef FRESCO_DELTACONV_CUH
#define FRESCO_DELTACONV_CUH

#include "Utils.cuh"
#include "NonZero.cuh"

namespace fresco {

    void deltaConv2d(const float *img, const float *img_prev, const float *ker, float *out, const float *out_prev,
                     int chan_in, int chan_out,
                     int img_w, int img_h,
                     int ker_w, int ker_h,
                     int stride_x, int stride_y,
                     int pad_x, int pad_y,
                     int dilation_x, int dilation_y);

}


#endif //FRESCO_DELTACONV_CUH
