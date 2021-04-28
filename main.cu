#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include "Conv.cuh"
#include "DeltaConv.cuh"
#include <fstream>
#include <string>
#include "Utils.cuh"
#include <curand.h>

using namespace std;
using namespace fresco;

unsigned long clock2ms(clock_t start, clock_t end) {
    return (end - start) * 1000 / CLOCKS_PER_SEC;
}

void randn(float *dest, int n) {
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, dest, n);
    curandDestroyGenerator(gen);
}

void sparse_copy(float *dest, float *src, int n, float p) {
    cudaMemcpy(dest, src, int((float) n * (1 - p)) * sizeof(float), cudaMemcpyDeviceToDevice);

}

int read_testdata(string filename, float *dest) {

    ifstream file(filename);
    string str;

    // read first line
    std::getline(file, str);

    std::cout << filename << std::endl;

    const int len = stoi(str);
    int i = 0;
    auto *buffer = (float *) malloc(len * sizeof(float));

    while (std::getline(file, str)) {
        buffer[i] = stof(str);
        ++i;
    }

    cudaMemcpy(dest, buffer, len * sizeof(float), cudaMemcpyHostToDevice);


    free(buffer);

    return len;
}

int main() {

    int img_w = 2048, img_h = 2048;

    int chan_in = 3;
    int chan_out = 4;

    int ker_w = 10, ker_h = 10;

    int stride_x = 1, stride_y = 1;

    int pad_x = 2, pad_y = 2;
    int dilation_x = 1, dilation_y = 1;

    int out_w = (img_w + 2 * pad_x - dilation_x * (ker_w - 1) - 1) / stride_x + 1;
    int out_h = (img_h + 2 * pad_y - dilation_y * (ker_h - 1) - 1) / stride_y + 1;

    float *img, *img_prev;
    float *out, *out_prev;
    float *ker;

    int img_size = img_h * img_w * chan_in;
    int out_size = out_h * out_w * chan_out;

    cout << "img size: " << img_size << " (w: " << img_w << ", h: " << img_h << ", chan: " << chan_in << ")" << endl;
    cout << "out size: " << out_size << " (w: " << out_w << ", h: " << out_h << ", chan: " << chan_out << ")" << endl;

    cudaMalloc(&img, img_size * sizeof(float));
    cudaMalloc(&img_prev, img_size * sizeof(float));

    cudaMalloc(&out, out_size * sizeof(float));
    cudaMalloc(&out_prev, out_size * sizeof(float));
    //cudaMalloc(&out_gt, out_size * sizeof(float));

    cudaMalloc(&ker, chan_in * ker_h * ker_w * chan_out * sizeof(float));

    // initialize prev
//
//    thrust::device_ptr<float> img_ptr(img);
//    thrust::fill(img_ptr, img_ptr + img_size, 1.0);

    randn(img, img_size);
    randn(ker, chan_in * ker_h * ker_w * chan_out);


    // create some duplications here

    // sparsity = 1 (all zero)
    // sparsity = 0 (all non-zero)
    sparse_copy(img_prev, img, img_size, 0);
//
//    thrust::device_ptr<float> img_prev_ptr(img_prev);
//    thrust::fill(img_prev_ptr, img_prev_ptr + img_size, 0.0);
//
//    thrust::device_ptr<float> out_prev_ptr(out_prev);
//    thrust::fill(out_prev_ptr, out_prev_ptr + out_size, 0.0);

//
//    thrust::device_ptr<float> dev_ptr2(img_prev);
//    thrust::fill(dev_ptr2 + (img_size / 2), dev_ptr2 + img_size, 1.0);
//    read_testdata("./testdata/img.txt", img);
//    read_testdata("./testdata/ker.txt", ker);
//    read_testdata("./testdata/out.txt", out_gt);

    // Warmup
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


    int repeat = 3;

    clock_t start = clock();
    for (int i = 0; i < repeat; i++) {
        fresco::deltaConv2d(
                img, img_prev,
                ker,
                out, out_prev,
                chan_in, chan_out,
                img_w, img_h,
                ker_w, ker_h,
                stride_x, stride_y,
                pad_x, pad_y,
                dilation_x, dilation_y, i == 0);
        //printData<float>(out, out_size, 30);
    }
    clock_t end = clock();

    cout << "delta conv: " << clock2ms(start, end) / repeat << " ms" << endl;

    start = clock();
    for (int i = 0; i < repeat; i++) {
        fresco::conv2d(
                img,
                ker,
                out,
                chan_in, chan_out,
                img_w, img_h,
                ker_w, ker_h,
                stride_x, stride_y,
                pad_x, pad_y,
                dilation_x, dilation_y, i == 0);
        //printData<float>(out, out_size, 30);
    }
    end = clock();
    cout << "conv: " << clock2ms(start, end) / repeat << " ms" << endl;

    cudaFree(img);
    cudaFree(out);
    cudaFree(img_prev);
    cudaFree(out_prev);
    cudaFree(ker);

    return 0;
}