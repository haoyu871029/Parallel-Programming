/* lab3_gpu_v2.cu big block version */

#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define MASK_N 2 //filter 數
#define MASK_X 5 
#define MASK_Y 5
#define xBound MASK_X/2
#define yBound MASK_Y/2
#define SCALE 8

__constant__ int deviceMask[MASK_N][MASK_X][MASK_Y];
/* Hint 7 */
// this variable is used by device
__constant__ char mask[MASK_N][MASK_Y][MASK_X] = { { { -1, -4, -6, -4, -1 },
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

int read_png(const char* filename, unsigned char** image, unsigned* height, unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb"); //使用 fopen 打開指定的 PNG 文件以便二進制讀取。
    if (infile == NULL) {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        return -1;
    }

    fread(sig, 1, 8, infile); //讀取文件的前 8 個字節
    if (!png_check_sig(sig, 8)) //檢查此 8 個字節是否為有效的 PNG 標誌。
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); //創建 PNG 讀取結構。
    if (!png_ptr) //如果內存不足，則返回錯誤。
        return 4;   /* out of memory */

    info_ptr = png_create_info_struct(png_ptr); //創建與 PNG 文件相關的信息結構
    if (!info_ptr) { //如果內存不足，則清理已創建的結構並返回錯誤。
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile); //初始化 PNG 讀取 - 設定 PNG 讀取函式的 I/O
    png_set_sig_bytes(png_ptr, 8); //初始化 PNG 讀取 - 設定 PNG 讀取函式的信號字節。
    png_read_info(png_ptr, info_ptr); //讀取 PNG 文件信息，PNG 信息結構被填充了來自文件的詳細信息。
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL); //獲取圖片屬性

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);  //更新 PNG 讀取信息
    rowbytes = png_get_rowbytes(png_ptr, info_ptr); //計算圖片每行的字節數
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    *image = (unsigned char *) malloc(rowbytes * *height); //根據圖片每行的字節數分配內存空間來存儲圖片數據
    if (image == NULL) { //如果內存分配失敗，則清理結構並返回錯誤。
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL); 
        return 3;
    }

    for (i = 0;  i < *height;  ++i) //為每一行圖片數據設置指針
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers); //讀取圖片數據到分配的內存中
    png_read_end(png_ptr, NULL); //結束圖片讀取
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, const unsigned channels) {

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

/* Hint 5 */
// this function is called by host and executed by device
__global__ void sobel (unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if(x >= width || y >= height) 
        return;

    float R_hor=0, G_hor=0, B_hor=0;
    float R_ver=0, G_ver=0, B_ver=0;

    #pragma unroll 5
    for (int v = -yBound; v <= yBound; ++v) {
        #pragma unroll 5
        for (int u = -xBound; u <= xBound; ++u) {
            if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                const unsigned char R = s[channels * (width * (y + v) + (x + u)) + 2];
                const unsigned char G = s[channels * (width * (y + v) + (x + u)) + 1];
                const unsigned char B = s[channels * (width * (y + v) + (x + u)) + 0];
                R_hor += R * mask[0][u + xBound][v + yBound];
                G_hor += G * mask[0][u + xBound][v + yBound];
                B_hor += B * mask[0][u + xBound][v + yBound];
                R_ver += R * mask[1][u + xBound][v + yBound];
                G_ver += G * mask[1][u + xBound][v + yBound];
                B_ver += B * mask[1][u + xBound][v + yBound];
            }
        }
    }

    float totalR = sqrt(R_hor * R_hor + R_ver * R_ver) / SCALE;
    float totalG = sqrt(G_hor * G_hor + G_ver * G_ver) / SCALE;
    float totalB = sqrt(B_hor * B_hor + B_ver * B_ver) / SCALE;
    const unsigned char cR = (totalR > 255.) ? 255 : totalR;
    const unsigned char cG = (totalG > 255.) ? 255 : totalG;
    const unsigned char cB = (totalB > 255.) ? 255 : totalB;
    t[channels * (width * y + x) + 2] = cR;
    t[channels * (width * y + x) + 1] = cG;
    t[channels * (width * y + x) + 0] = cB;
}

int main(int argc, char** argv) {

    assert(argc == 3);

    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    cudaHostRegister(host_s, height * width * channels * sizeof(unsigned char), cudaHostRegisterDefault);
    
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst
    unsigned char* device_s;
    cudaMalloc((void **)&device_s, height * width * channels * sizeof(unsigned char));
    unsigned char* device_t;
    cudaMalloc((void **)&device_t, height * width * channels * sizeof(unsigned char));
    
    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(device_s, host_s, height * width * channels * sizeof(unsigned char), cudaMemcpyHostToDevice);
    
    /* Hint 3 */
    // acclerate this function
    dim3 dimGrid(ceil(width/32.0), ceil(height/32.0), 1); //block: (行,列)
    dim3 dimBlock(32, 32, 1); //thread: (行,列)
    sobel<<<dimGrid, dimBlock>>>(device_s, device_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    cudaMemcpy(host_t, device_t, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    
    write_png(argv[2], host_t, height, width, channels);

    /* Free device memory */

    cudaFree(device_s);
    cudaFree(device_t);

    /* Free host memory */

    free(host_s);
    free(host_t);

    return 0;
}

/* compile & execute */

//in hades01 (因為 hades02 無法編譯、lab4-judge、pnd-diff，只能執行以編譯好的程式)
//testcases: (candy.out.png  candy.png  candy.txt  jerry.out.png  jerry.png  jerry.txt  large-candy.out.png  large-candy.png  large-candy.txt)

//(copy this code to lab4.cu)
//compile: nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -lpng -lz -o lab4 lab4.cu
//execute: time srun -n 1 --gres=gpu:1 ./lab4 /home/pp23/share/lab4/testcases/large-candy.png large-candy.out.png
//validate: png-diff large-candy.out.png /home/pp23/share/lab4/testcases/large-candy.out.png
//judge: lab4-judge