#include <iostream>

#include "PrefixSum.cuh"
#include "StreamIndexCompaction.cuh"
#include <cublas_v2.h>

const int NUM_THREADS = 1024;

inline int getNumBlock(int n, int threads) {
    return (n - 1) / threads + 1;
}


__global__ void
frameDifferencing(const float *currImg, const float *prevImg, float *diffImg, bool *idMapImg, int chan, int bound) {
    // img layout (H, W, C)
    int offset = threadIdx.x + blockDim.x * blockIdx.x;
    if (offset < bound) {

        int offset_img = offset * chan;
        bool is_id = true;

        // consecutive access to the buffer... so I hope it utilizes the cache well.
        for (int i = 0; i < chan; i++) {
            int idx = i + offset_img;
            float d = currImg[idx] - prevImg[idx];

            diffImg[idx] = d;


            // epsilon. may slightly differ from context to context.
            if (d > 0.001) {
                is_id = false;
            }
        }

        idMapImg[offset] = is_id;
    }
}

__global__ void convolveIdMap(bool *idMapImg, bool *idMapConvImg,
                              int w, int h,
                              int conv_w, int conv_h,
                              int ker_w, int ker_h,
                              int stride_x, int stride_y,
                              int pad_x, int pad_y,
                              int dilation_w, int dilation_h) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < conv_w * conv_h) {

        int x = idx % blockDim.x;
        int y = idx / blockDim.x;

        int offset_x = x * stride_x - pad_x;
        int offset_y = y * stride_y - pad_y;

        int count = 0;

        // we may use some relaxed threshold other than 1, (e.g., conv_w * conv_h / 10)
        int count_thres = 1;

        bool exit_loop = false;

        for (int i = 0; i < ker_h; ++i) {
            int ker_y = i * dilation_h + offset_y;
            for (int j = 0; j < ker_w; ++j) {
                int ker_x = j * dilation_w + offset_x;
                if (ker_y >= 0 && ker_x >= 0 && ker_y < h && ker_x < w) {
                    if (idMapImg[ker_y * w + ker_x]) {
                        count++;
                        if (count >= count_thres) {
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
        idMapConvImg[y * conv_w + x] = count >= count_thres;
    }
}

__global__ void im2scol(int *conv_indices, float *diffImg,
                        float *scol_out,
                        int num_chan,
                        int w, int h,
                        int conv_w, int conv_h,
                        int ker_w, int ker_h,
                        int stride_x, int stride_y,
                        int pad_x, int pad_y,
                        int dilation_w, int dilation_h) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int chan_idx = idx % num_chan;

    int t_idx = idx / num_chan;

    if (t_idx < conv_w * conv_h) {

        int conv_idx = conv_indices[t_idx];

        int conv_x = conv_idx % conv_w;
        int conv_y = conv_idx / conv_w;

        int img_x_offset = conv_x * stride_x - pad_x;
        int img_y_offset = conv_y * stride_y - pad_y;

        int scol_idx_offset = t_idx * num_chan * ker_w * ker_h + chan_idx * (ker_w * ker_h);

        for (int ker_y = 0; ker_y < ker_h; ++ker_y) {
            int img_y = ker_y * dilation_h + img_y_offset;
            for (int ker_x = 0; ker_x < ker_w; ++ker_x) {
                int img_x = ker_x * dilation_w + img_x_offset;
                if (img_y >= 0 && img_x >= 0 && img_y < h && img_x < w) {

                    int img_idx = img_y * w * num_chan + img_x * num_chan + chan_idx;

                    int scol_idx = scol_idx_offset + ker_y * ker_w + ker_x;

                    scol_out[scol_idx] = diffImg[img_idx];
                }
            }
        }


    }


}


// Multiply the arrays A and B on GPU and save the result in C
// C(m,n) = A(m,k) * B(k,n)
void gemm(const float *A, const float *B, float *C, const int m, const int k, const int n) {
    int lda = m, ldb = k, ldc = m;
    const float alf = 1;
    const float bet = 0;
    const float *alpha = &alf;
    const float *beta = &bet;

    // Create a handle for CUBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Do the actual multiplication
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);

    // Destroy the handle
    cublasDestroy(handle);
}

// column-wise, accumulates all channels to the destination image.
__global__ void
scol2im(float *scol, int *conv_indices, int conv_indices_len, float *base_img, int scol_num_chan, int w) {

    int idx = threadIdx.x + blockDim.x * blockIdx.x;


    if (idx < conv_indices_len) {

        int img_idx = conv_indices[idx];

        int img_y = img_idx / w;
        int img_x = img_idx % w;

        // accumulate all channel values to the destination.

        int img_offset = img_y * w * scol_num_chan + img_x * scol_num_chan;
        int scol_offset = idx * scol_num_chan;

        for (int i = 0; i < scol_num_chan; ++i) {
            base_img[img_offset + i] += scol[scol_offset + i];
        }
    }

}

int main() {

    int imgInW = 1024, imgInH = 1024;
    int imgInChan = 3;
    int imgOutChan = 4;
    int kerW = 8, kerH = 8;
    int strideX = 1, strideY = 1;
    int padX = 0, padY = 0;
    int dilationX = 1, dilationY = 1;

    int imgOutW = (imgInW + 2 * padX - dilationX * (kerW - 1) - 1) / strideX;
    int imgOutH = (imgInH + 2 * padY - dilationY * (kerH - 1) - 1) / strideY;


    float *prevImgIn, *prevImgOut;
    float *currImgIn, *currImgOut;

    cudaMalloc(&prevImgIn, imgInH * imgInW * imgInChan * sizeof(float));
    cudaMalloc(&currImgIn, imgInH * imgInW * imgInChan * sizeof(float));
    cudaMalloc(&prevImgOut, imgOutH * imgOutW * imgOutChan * sizeof(float));
    cudaMalloc(&currImgOut, imgOutH * imgOutW * imgOutChan * sizeof(float));

    // do some cudaMemcpy() stuff here...

    // Define some convolution kernels here

    float *convKernel;
    cudaMalloc(&convKernel, imgInChan * kerH * kerW * imgOutChan * sizeof(float));



    // STEP 1: Frame differencing

    // diffImg (imgInH, imgInW, imgInChan)
    // currImg - affineTransform(prevImg) 의 결과. 여기서 prevImg에 적용되는 아핀변환은 diffImg의 sparsity를 최대화할 수 있어야 함.

    float *diffImg;
    bool *idMapImg; // dim [H, W] | 모든 채널이 같으면 0 ('같다'의 기준을 조금 널널하게 할 수도 있을듯), 하나라도 다르면 1.

    cudaMalloc(&diffImg, imgInH * imgInW * imgInChan * sizeof(float));
    cudaMalloc(&idMapImg, imgInH * imgInW * sizeof(bool));

    frameDifferencing<<<getNumBlock(imgInH * imgInW, NUM_THREADS), NUM_THREADS>>>(currImgIn, prevImgIn, diffImg,
                                                                                  idMapImg,
                                                                                  imgInChan, imgInH * imgInW);


    // STEP 2: Apply max convolution to idMapImg

    bool *idMapConvImg;
    cudaMalloc(&idMapConvImg, imgOutH * imgOutW * sizeof(bool));


    convolveIdMap<<<getNumBlock(imgOutH * imgOutW, NUM_THREADS), NUM_THREADS>>>(idMapImg, idMapConvImg, imgInW, imgInH,
                                                                                imgOutW, imgOutH, kerW, kerH, strideX,
                                                                                strideY,
                                                                                padX, padY, dilationX, dilationY);


    // STEP 3: Convert idMapConvImg into sparse form

    int *indices;
    cudaMalloc(&indices, imgOutH * imgOutW * sizeof(int));

    int num_indices = get_nonzero_indices(idMapConvImg, indices, imgOutH * imgOutW, warpSize * 4);

    printf("%i", num_indices);


    // STEP 4: image to sparse column
    float *s_col;
    cudaMalloc(&s_col, num_indices * imgInChan * kerW * kerH * sizeof(float));

    im2scol<<<getNumBlock(num_indices * imgInChan, NUM_THREADS), NUM_THREADS>>>(indices, diffImg,
                                                                                s_col,
                                                                                imgInChan,
                                                                                imgInW, imgInH,
                                                                                imgOutW, imgOutH, kerW, kerH, strideX,
                                                                                strideY,
                                                                                padX, padY, dilationX, dilationY);

    // STEP 5: (Dense) GEMM

    float *s_img;
    cudaMalloc(&s_img, num_indices * imgOutChan * sizeof(float));

    gemm(s_col, convKernel, s_img, num_indices, imgInChan * kerH * kerW, imgOutChan);


    // STEP 6: sparse column to image
    // Apply (inline) affine transform to prevImgOut... but right now we are just doing manual blit.
    cudaMemcpy(currImgOut, prevImgOut, imgOutW * imgOutH * imgOutChan, cudaMemcpyKind::cudaMemcpyDeviceToDevice);

    scol2im<<<getNumBlock(num_indices, NUM_THREADS), NUM_THREADS>>>(s_img, indices, num_indices, currImgOut, imgOutChan,
                                                                    imgOutW);


    // Free memory
    cudaFree(prevImgIn);
    cudaFree(currImgIn);
    cudaFree(prevImgOut);
    cudaFree(currImgOut);
    cudaFree(convKernel);


    cudaFree(diffImg);
    cudaFree(idMapImg);
    cudaFree(idMapConvImg);
    cudaFree(indices);

    return 0;
}