/* lab4 source code by TA */

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define Z 2 //MASK_N
#define Y 5 //MASK_Y
#define X 5 //MASK_X
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}
void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}
__constant__ char mask[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {

    /* thread indexing */
    
    int y = blockIdx.x * blockDim.x + threadIdx.x; //(y,0) 代表的是 thread 在全部 threads 中的編號，也代表此條 thread 要處理第幾列的資料
    if (y >= height) 
        return;

    double val[Z][3];
    for (int x = 0; x < width; ++x) {//代表每條 thread 處理一列 pixels
        for (int i = 0; i < Z; ++i) {//共兩圈，一圈處理水平filter，一圈處理垂直filter

            /* 處理此 pixel 的初始 RGB 計算值 */

            val[i][2] = 0.;
            val[i][1] = 0.;
            val[i][0] = 0.;

            /* 經過相鄰 pixels 的參與，算出該 pixel 的「RGB水平計算值」和「RGB垂直計算值」
               以 row major 去 index 相鄰 pixels */
            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {//符合條件表示該相鄰 pixel 在圖片範圍內
                        const unsigned char R = s[channels * (width * (y + v) + (x + u)) + 2];
                        const unsigned char G = s[channels * (width * (y + v) + (x + u)) + 1];
                        const unsigned char B = s[channels * (width * (y + v) + (x + u)) + 0];
                        val[i][2] += R * mask[i][u + xBound][v + yBound];
                        val[i][1] += G * mask[i][u + xBound][v + yBound];
                        val[i][0] += B * mask[i][u + xBound][v + yBound];
                    }
                }
            }
        }

        /* 處理此 pixel 的最終 RGB 計算值 */

        double totalR = 0.;
        double totalG = 0.;
        double totalB = 0.;
        for (int i = 0; i < Z; ++i) {
            totalR += val[i][2] * val[i][2];
            totalG += val[i][1] * val[i][1];
            totalB += val[i][0] * val[i][0];
        }

        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;

        const unsigned char cR = (totalR > 255.) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.) ? 255 : totalB;

        /* 將處理後的最終 RGB 計算值寫入代表該 pixel 的 3 個記憶體位置 */

        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;
    }
}

int main(int argc, char **argv) {
    
    assert(argc == 3);

    /* read the image to src, and get height, width, channels */

    unsigned height, width, channels;
    unsigned char *src = NULL;
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }
    cudaHostRegister(src, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);

    /* cudaMalloc(...) for device src and device dst */

    unsigned char *dsrc, *ddst;
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));

    /* cudaMemcpy(...) copy source image to device (mask matrix if necessary) */

    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);

    /* decide to use how many blocks and threads & launch cuda kernel */
    // acclerate this function

    const int num_threads = 256; //dimBlock
    const int num_blocks = height / num_threads + 1; //dimGrid
    sobel << <num_blocks, num_threads>>> (dsrc, ddst, height, width, channels);

    /* cudaMemcpy(...) copy result image to host */

    unsigned char *dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    cudaMemcpy(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    /* write PNG */

    write_png(argv[2], dst, height, width, channels);

    /* Free host memory & device memory */

    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);

    return 0;
}

/* compile & execute */

//in hades01 (因為 hades02 無法編譯、lab4-judge、pnd-diff，只能執行以編譯好的程式)
//testcases: (candy.out.png  candy.png  candy.txt  jerry.out.png  jerry.png  jerry.txt  large-candy.out.png  large-candy.png  large-candy.txt)

//(copy this code to lab4.cu)
//compile: nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -lpng -lz -o lab4 lab4.cu
//execute: time srun -n 1 --gres=gpu:1 ./lab4 /home/pp23/share/lab4/testcases/large-candy.png large-candy.out.png
//validate: png-diff large-candy.out.png /home/pp23/share/lab3/testcases/large-candy.out.png
//judge: lab4-judge