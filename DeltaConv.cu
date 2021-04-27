//
// Created by hyunjk on 4/27/21.
//
#include <iostream>
#include "DeltaConv.cuh"

namespace fresco {

    __global__ void frameDifferencing(const float *img_curr, const float *img_prev,
                                      float *img_out, bool *d_map,
                                      const int chan,
                                      const int img_w, const int img_h,
                                      const float eps = 0.01) {
        // img layout (H, W, C)
        // t_i: 0 -> H * W
        const int t_i = threadIdx.x + blockDim.x * blockIdx.x;

        if (t_i >= img_h * img_w) {
            return;
        }

        const int img_idx_offset = t_i * chan;
        bool is_diff = false;

        // consecutive access to the buffer... so I hope it utilizes the cache well.
        for (int i = 0; i < chan; i++) {
            const int img_idx = i + img_idx_offset;
            const float delta = img_curr[img_idx] - img_prev[img_idx];

            img_out[img_idx] = delta;


            // epsilon. may slightly differ from context to context.
            if (delta > eps) {
                is_diff = true;
            }
        }

        d_map[t_i] = is_diff;
    }


    __global__ void convDmap(const bool *d_map, bool *d_map_conv,
                             const int img_w, const int img_h,
                             const int out_w, const int out_h,
                             const int ker_w, const int ker_h,
                             const int stride_x, const int stride_y,
                             const int pad_x, const int pad_y,
                             const int dilation_x, const int dilation_y,
                             const int thres = 1) {

        // t_i: 0 -> out_w * out_h
        const int t_i = threadIdx.x + blockDim.x * blockIdx.x;

        if (t_i >= out_w * out_h) {
            return;
        }

        const int out_x = t_i % out_w;
        const int out_y = t_i / out_w;

        const int img_x_offset = out_x * stride_x - pad_x;
        const int img_y_offset = out_y * stride_y - pad_y;

        int count = 0;
        bool exit_loop = false;

        for (int ker_y = 0; ker_y < ker_h; ++ker_y) {

            const int img_y = ker_y * dilation_y + img_y_offset;

            for (int ker_x = 0; ker_x < ker_w; ++ker_x) {

                const int img_x = ker_x * dilation_x + img_x_offset;

                if (img_y >= 0 && img_x >= 0 && img_y < img_h && img_x < img_w) {

                    const int img_idx = img_y * img_w + img_x;

                    if (d_map[img_idx]) {
                        count++;
                        if (count >= thres) {
                            exit_loop = true;
                            break;
                        }

                    }
                }
            }
            if (exit_loop) {
                break;
            }
        }

        const int out_idx = out_y * out_w + out_x;

        d_map_conv[out_idx] = count >= thres;

    }


    __global__ void im2colIndexed(const int *col_indices, const int col_indices_len,
                                  const float *img, float *col_indexed,
                                  const int chan,
                                  const int img_w, const int img_h,
                                  const int col_w, const int col_h,
                                  const int ker_w, const int ker_h,
                                  const int stride_x, const int stride_y,
                                  const int pad_x, const int pad_y,
                                  const int dilation_x, const int dilation_y) {

        // t_i: 0 -> col_indices_len * chan
        int t_i = threadIdx.x + blockDim.x * blockIdx.x;
        int chan_i = t_i % chan;
        int col_i_i = t_i / chan;

        if (col_i_i >= col_indices_len) {
            return;
        }

        int col_i = col_indices[col_i_i];

        int col_x = col_i % col_w;
        int col_y = col_i / col_w;

        int img_x_offset = col_x * stride_x - pad_x;
        int img_y_offset = col_y * stride_y - pad_y;

        // col_i_i, not col_i!
        int col_idx_offset = (col_i_i * chan + chan_i) * (ker_w * ker_h);

        for (int ker_y = 0; ker_y < ker_h; ++ker_y) {

            int img_y = ker_y * dilation_y + img_y_offset;

            for (int ker_x = 0; ker_x < ker_w; ++ker_x) {

                int img_x = ker_x * dilation_x + img_x_offset;

                if (img_y >= 0 && img_x >= 0 && img_y < img_h && img_x < img_w) {

                    int img_idx = img_y * img_w * chan + img_x * chan + chan_i;
                    int col_idx = col_idx_offset + ker_y * ker_w + ker_x;
                    col_indexed[col_idx] = img[img_idx];
                }
            }
        }
    }


    // column-wise, accumulates all channels to the destination image.
    __global__ void unfoldIndexedImg(const int *img_indices, const int img_indices_len,
                                     const float *img_indexed, float *img_base,
                                     int chan,
                                     int img_w, int img_h) {

        // t_i: 0 -> col_indices_len
        int t_i = threadIdx.x + blockDim.x * blockIdx.x;

        if (t_i >= img_indices_len) {
            return;
        }

        int img_i = img_indices[t_i];

        int img_x = img_i % img_w;
        int img_y = img_i / img_w;

        int img_idx_offset = (img_y * img_w + img_x) * chan;
        int img_indexed_idx_offset = t_i * chan;

        for (int i = 0; i < chan; ++i) {
            img_base[img_idx_offset + i] += img_indexed[img_indexed_idx_offset + i];
        }
    }


    // [H, W, C] -> [OH, OW, C, KH, KW]
    void deltaConv2d(const float *img, const float *img_prev,
                     const float *ker,
                     float *out, const float *out_prev,
                     int chan_in, int chan_out,
                     int img_w, int img_h,
                     int ker_w, int ker_h,
                     int stride_x, int stride_y,
                     int pad_x, int pad_y,
                     int dilation_x, int dilation_y) {


        int out_w = (img_w + 2 * pad_x - dilation_x * (ker_w - 1) - 1) / stride_x;
        int out_h = (img_h + 2 * pad_y - dilation_y * (ker_h - 1) - 1) / stride_y;

        // ************************************
        // STEP 1: Frame differencing
        // ************************************

        // img_diff (imgInH, imgInW, imgInChan)
        // currImg - affineTransform(prevImg) 의 결과. 여기서 prevImg에 적용되는 아핀변환은 diffImg의 sparsity를 최대화할 수 있어야 함.

        float *img_diff;
        bool *d_map; // dim [H, W] | 모든 채널이 같으면 0 ('같다'의 기준을 조금 널널하게 할 수도 있을듯), 하나라도 다르면 1.
        cudaMalloc(&img_diff, img_h * img_w * chan_in * sizeof(float));
        cudaMalloc(&d_map, img_h * img_w * sizeof(bool));

        frameDifferencing<<<getNumBlock(img_h * img_w, NUM_THREADS), NUM_THREADS>>>(
                img, img_prev,
                img_diff, d_map,
                chan_in,
                img_w, img_h);

        // ************************************
        // STEP 2: Apply max convolution to d_map
        // ************************************

        bool *out_d_map;
        cudaMalloc(&out_d_map, out_h * out_w * sizeof(bool));

        convDmap<<<getNumBlock(out_h * out_w, NUM_THREADS), NUM_THREADS>>>(
                d_map, out_d_map,
                img_w, img_h,
                out_w, out_h,
                ker_w, ker_h,
                stride_x, stride_y,
                pad_x, pad_y,
                dilation_x, dilation_y);

        // ************************************
        // STEP 3: Convert out_d_map into sparse form
        // ************************************

        int *out_indices;
        cudaMalloc(&out_indices, out_h * out_w * sizeof(int));

        int out_indices_len = get_nonzero_indices(out_d_map, out_indices, out_h * out_w, 32 * 4);

        // exact same current image and prev image
        if (out_indices_len == 0) {

            cudaFree(img_diff);
            cudaFree(d_map);
            cudaFree(out_d_map);
            cudaFree(out_indices);

            cudaMemcpy(out, out_prev, out_h * out_w * chan_out, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
            return;
        }

        // ************************************
        // STEP 4: image to sparse column
        // ************************************

        float *col_indexed;
        cudaMalloc(&col_indexed, out_indices_len * chan_in * ker_h * ker_w * sizeof(float));

        im2colIndexed<<<getNumBlock(out_indices_len * chan_in, NUM_THREADS), NUM_THREADS>>>(
                out_indices, out_indices_len,
                img_diff, col_indexed,
                chan_in,
                img_w, img_h,
                out_w, out_h,
                ker_w, ker_h,
                stride_x, stride_y,
                pad_x, pad_y,
                dilation_x, dilation_y);


        // ************************************
        // STEP 5: (Dense) GEMM
        // ************************************


        float *img_indexed;
        cudaMalloc(&img_indexed, out_indices_len * chan_out * sizeof(float));

        gemm(col_indexed, ker, img_indexed, out_indices_len, chan_in * ker_h * ker_w, chan_out);


        // ************************************
        // STEP 6: sparse column to image
        // ************************************

        // Apply (inline) affine transform to prevImgOut... but right now we are just doing manual blit.
        cudaMemcpy(out, out_prev, out_h * out_w * chan_out, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

        unfoldIndexedImg<<<getNumBlock(out_indices_len, NUM_THREADS), NUM_THREADS>>>(
                out_indices, out_indices_len,
                img_indexed,
                out,
                chan_out,
                out_w, out_h);

        // free all allocated resources
        cudaFree(img_diff);
        cudaFree(d_map);
        cudaFree(out_d_map);
        cudaFree(out_indices);
        cudaFree(col_indexed);
        cudaFree(img_indexed);
    }


}