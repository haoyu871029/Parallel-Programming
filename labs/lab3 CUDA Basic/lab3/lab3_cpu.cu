/* lab3 CPU version */
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <zlib.h>
#include <png.h>

#define MASK_N 2 //number of filter matrixs
#define MASK_X 5
#define MASK_Y 5
#define SCALE 8

/* Hint 7 */
// this variable is used by device
int mask[MASK_N][MASK_X][MASK_Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

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
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL); //height, width, channels 都在 png_get_IHDR 被設定好了

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

/* Hint 5 */
// this function is called by host and executed by device
void sobel (unsigned char* s, unsigned char* t, unsigned height, unsigned width, unsigned channels) {
    int  x, y, i, v, u;
    int  R, G, B; //channels
    double val[MASK_N*3] = {0.0};
    /* val size 為 2*3，2 代表水平和垂直兩個 filter，3 代表 RGB
       val用來存該pixel的「RGB水平計算值」和「RGB垂直計算值」
       val[0,1,2] 分別用來存 BGR 水平計算值
       val[3,4,5] 分別用來存 BGR 垂直計算值 */

    int adjustX, adjustY, xBound, yBound;
    adjustX = (MASK_X % 2) ? 1 : 0;
    adjustY = (MASK_Y % 2) ? 1 : 0;
    xBound = MASK_X /2;
    yBound = MASK_Y /2;

    /* Hint 6 */
    // parallel job by blockIdx, blockDim, threadIdx 
    for (y = 0; y < height; ++y) {//y 代表所在列
        for (x = 0; x < width; ++x) {//x 代表所在行

            /* 處理此 pixel 的初始 RGB 計算值 */

            for (i = 0; i < MASK_N; ++i) {//共兩圈，一圈處理水平filter，一圈處理垂直filter

                /* pixel_value initialization */

                val[i*3+2] = 0.0;
                val[i*3+1] = 0.0;
                val[i*3] = 0.0;

                /* 經過相鄰 pixels 的參與，算出該 pixel 的「RGB水平計算值」和「RGB垂直計算值」
                   以 row major 去 index 相鄰 pixels */
                for (v = -yBound; v < yBound + adjustY; ++v) {
                    for (u = -xBound; u < xBound + adjustX; ++u) {
                        if ((x + u) >= 0 && (x + u) < width && (y + v) >= 0 && (y + v) < height) {//符合條件表示該相鄰 pixel 在圖片範圍內
                            R = s[channels * (width * (y+v) + (x+u)) + 2];
                            G = s[channels * (width * (y+v) + (x+u)) + 1];
                            B = s[channels * (width * (y+v) + (x+u)) + 0];
                            val[i*3+2] += R * mask[i][u + xBound][v + yBound];
                            val[i*3+1] += G * mask[i][u + xBound][v + yBound];
                            val[i*3+0] += B * mask[i][u + xBound][v + yBound];
                        }    
                    }
                }
            }

            /* 處理此 pixel 的最終 RGB 計算值 */

            double totalR = 0.0;
            double totalG = 0.0;
            double totalB = 0.0;
            for (i = 0; i < MASK_N; ++i) {
                totalR += val[i * 3 + 2] * val[i * 3 + 2];
                totalG += val[i * 3 + 1] * val[i * 3 + 1];
                totalB += val[i * 3 + 0] * val[i * 3 + 0];
            }

            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            
            const unsigned char cR = (totalR > 255.0) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.0) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.0) ? 255 : totalB;

            /* 將處理後的最終 RGB 計算值寫入代表該 pixel 的 3 個記憶體位置 */

            t[channels * (width * y + x) + 2] = cR;
            t[channels * (width * y + x) + 1] = cG;
            t[channels * (width * y + x) + 0] = cB;
        }
    }
}

int main(int argc, char** argv) {

    assert(argc == 3);

    unsigned height, width, channels;
    unsigned char* host_s = NULL;
    read_png(argv[1], &host_s, &height, &width, &channels);
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));
    
    /* Hint 1 */
    // cudaMalloc(...) for device src and device dst

    /* Hint 2 */
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)

    /* Hint 3 */
    // acclerate this function
    sobel(host_s, host_t, height, width, channels);
    
    /* Hint 4 */
    // cudaMemcpy(...) copy result image to host
    
    write_png(argv[2], host_t, height, width, channels);

    return 0;
}

/* compile & execute */

//in hades01 (因為 hades02 無法編譯、lab3-judge、pnd-diff，只能執行以編譯好的程式)
//testcases: (candy.out.png  candy.png  candy.txt  jerry.out.png  jerry.png  jerry.txt  large-candy.out.png  large-candy.png  large-candy.txt)

//(copy this code to lab3.cu)
//compile: nvcc  -O3 -std=c++11 -Xptxas=-v -arch=sm_61 -lpng -lz -o lab3 lab3.cu
//execute: time srun -n 1 --gres=gpu:1 ./lab3 /home/pp23/share/lab3/testcases/large-candy.png large-candy.out.png
//validate: png-diff large-candy.out.png /home/pp23/share/lab3/testcases/large-candy.out.png
//judge: lab3-judge