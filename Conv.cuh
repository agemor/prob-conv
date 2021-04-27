//
// Created by hyunjk on 4/27/21.
//

#ifndef FRESCO_CONV_CUH
#define FRESCO_CONV_CUH

#include "Utils.cuh"

namespace fresco {

    void conv2d(const float *img, const float *ker, float *out,
                int chan_in, int chan_out,
                int img_w, int img_h,
                int ker_w, int ker_h,
                int stride_x, int stride_y,
                int pad_x, int pad_y,
                int dilation_x, int dilation_y);

}
#endif //FRESCO_CONV_CUH
