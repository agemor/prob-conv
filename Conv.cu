//
// Created by hyunjk on 4/27/21.
//

#include "Conv.cuh"

namespace fresco {


    // [H, W, C] -> [OH, OW, C, KH, KW]
    __global__ void im2col(const float *img, float *col,
                           const int chan,
                           const int img_w, const int img_h,
                           const int col_w, const int col_h,
                           const int ker_w, const int ker_h,
                           const int stride_x, const int stride_y,
                           const int pad_x, const int pad_y,
                           const int dilation_x, const int dilation_y) {

        // 각 thread는 채널 하나에서의 conv연산을 맡는다.
        // N = col_h * col_w * chan
        const int t_i = threadIdx.x + blockDim.x * blockIdx.x;
        const int chan_i = t_i % chan;
        const int col_i = t_i / chan;

        if (col_i >= col_w * col_h) {
            return;
        }

        const int col_x = col_i % col_w;
        const int col_y = col_i / col_w;

        const int img_x_offset = col_x * stride_x - pad_x;
        const int img_y_offset = col_y * stride_y - pad_y;

        const int col_idx_offset = (col_i * chan + chan_i) * (ker_w * ker_h);

        for (int ker_y = 0; ker_y < ker_h; ++ker_y) {

            int img_y = ker_y * dilation_y + img_y_offset;

            for (int ker_x = 0; ker_x < ker_w; ++ker_x) {

                int img_x = ker_x * dilation_x + img_x_offset;
                float val = 0.0;

                if (img_y >= 0 && img_x >= 0 && img_y < img_h && img_x < img_w) {
                    int img_idx = img_y * img_w * chan + img_x * chan + chan_i;
                    val = img[img_idx];
                }
                int col_idx = col_idx_offset + ker_y * ker_w + ker_x;
                col[col_idx] = val;
            }
        }
    }


    // img [H_in, W_in, C_in]
    // ker [C_in, KH, KW, C_out]
    // out [H_out, W_out, C_out]
    void conv2d(const float *img,
                const float *ker,
                float *out,
                int chan_in, int chan_out,
                int img_w, int img_h,
                int ker_w, int ker_h,
                int stride_x, int stride_y,
                int pad_x, int pad_y,
                int dilation_x, int dilation_y) {


        int out_w = (img_w + 2 * pad_x - dilation_x * (ker_w - 1) - 1) / stride_x + 1;
        int out_h = (img_h + 2 * pad_y - dilation_y * (ker_h - 1) - 1) / stride_y + 1;

        // STEP 1. im2col
        float *col; // [H_out, W_out, C_in, KH, KW]
        cudaMalloc(&col, out_h * out_w * chan_in * ker_h * ker_w * sizeof(float));

        int num_tasks = out_h * out_w * chan_in;
        im2col<<<getNumBlock(num_tasks, NUM_THREADS), NUM_THREADS>>>(img, col, chan_in, img_w, img_h, out_w, out_h,
                                                                     ker_w, ker_h, stride_x, stride_y, pad_x, pad_y,
                                                                     dilation_x, dilation_y);

        //std::cout << "im2col" << std::endl;
        //printData<float>(col, out_h * out_w * chan_in * ker_h * ker_w, 30);

        std::cout << "im2col size: " << out_h * out_w * chan_in * ker_h * ker_w << " (col: " << out_h * out_w << ", ker: " << ker_h * ker_w << ", chan: " << chan_in << ")" << std::endl;

        // STEP 2. gemm
        gemm(col, ker, out, out_h * out_w, chan_in * ker_h * ker_w, chan_out);

        // free all allocated resources
        cudaFree(col);
    }


}