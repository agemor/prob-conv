
#include "Conv.cuh"
#include "DeltaConv.cuh"

int main() {

    int img_w = 1024, img_h = 1024;
    int chan_in = 3;
    int chan_out = 4;
    int ker_w = 8, ker_h = 8;
    int stride_x = 1, stride_y = 1;
    int pad_x = 0, pad_y = 0;
    int dilation_x = 1, dilation_y = 1;

    int out_w = (img_w + 2 * pad_x - dilation_x * (ker_w - 1) - 1) / stride_x;
    int out_h = (img_h + 2 * pad_y - dilation_y * (ker_h - 1) - 1) / stride_y;

    float *img, *img_prev;
    float *out, *out_prev;
    float *ker;

    cudaMalloc(&img, img_h * img_w * chan_in * sizeof(float));
    cudaMalloc(&out, img_h * img_w * chan_in * sizeof(float));
    cudaMalloc(&img_prev, out_h * out_w * chan_out * sizeof(float));
    cudaMalloc(&out_prev, out_h * out_w * chan_out * sizeof(float));
    cudaMalloc(&ker, chan_in * ker_h * ker_w * chan_out * sizeof(float));

    fresco::conv2d(
            img,
            ker,
            out,
            chan_in, chan_out,
            img_w, img_h,
            ker_w, ker_h,
            stride_x, stride_y,
            pad_x, pad_y,
            dilation_x, dilation_y);

    fresco::deltaConv2d(
            img, img_prev,
            ker,
            out, out_prev,
            chan_in, chan_out,
            img_w, img_h,
            ker_w, ker_h,
            stride_x, stride_y,
            pad_x, pad_y,
            dilation_x, dilation_y);

    cudaFree(img);
    cudaFree(out);
    cudaFree(img_prev);
    cudaFree(out_prev);
    cudaFree(ker);


    return 0;
}