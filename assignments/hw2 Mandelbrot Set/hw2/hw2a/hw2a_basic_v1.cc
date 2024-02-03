#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

int thread_num, iters, width, height;
int* image;
double left, right, lower, upper;

void* work(void* tid){
/* mandelbrot set */

    int* id = (int*) tid;
    double range_per_unit_w = (right-left)/width; //每個 pixel 之間在 x 軸上跨多少 (寬度差)
    double range_per_unit_h = (upper-lower)/height; //每個 pixel 之間在 y 軸上跨多少 (高度差)

    for (int j=0; j<height; j++) {
    //一圈處理一列
        double b0 = lower + (j * range_per_unit_h); //此列的所有 pixel 在複座標系上的縱座標值
        for (int i=*id; i<width; i+=thread_num) {
        //ex. 10x10 image, 4 threads, thread 0: 0 4 8, thread 1: 1 5 9
            double a0 = left + (i * range_per_unit_w); //此列該 pixel 在複座標系上的橫座標值
            //this pixel's xy座標為 (i,j), 複座標為 (a0,b0)
            //this pixel is C = a0 + b0i
            int pixel_value = 0;
            double a = 0, b = 0, length_squared = 0;
            while (pixel_value < iters && length_squared < 4) {
                double temp = a*a - b*b + a0;
                b = 2*a*b + b0;
                a = temp;
                length_squared = a*a + b*b; //即 |Zk|^2
                pixel_value++;
            }
            image[j*width+i] = pixel_value;
            //so if C belongs to the Mandelbrot set, pixel_value = iters
            //else pixel_value < iters
        }
    }
    pthread_exit(NULL);
}
void write_png(const char* filename, int iters, int width, int height, const int* buffer) {
/* 將一個圖片緩衝區（buffer）寫入一個 PNG 圖片文件。 */
//write_png(filename, iters, width, height, image);

    /* 設定PNG */

    FILE* fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    /* 將一個由整數陣列 buffer 表示的圖像資料，逐行寫入至一個 PNG 文件。 */

    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    for (int y = 0; y < height; ++y) {
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set); //取得可用的core數
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    thread_num = CPU_COUNT(&cpu_set);
    pthread_t threads[thread_num];
    int tids[thread_num];

    /* argument parsing */
    //srun -n $procs -c $t ./exe $out $iter $a $x1 $x1 $y1 $w $h
    //image size (total pixels) is $w * $h

    assert(argc == 9);
    const char* filename = argv[1];    //$out
    iters  = strtol(argv[2], 0, 10);   //$iter
    left   = strtod(argv[3], 0);       //$a
    right  = strtod(argv[4], 0);       //$x1
    lower  = strtod(argv[5], 0);       //$x1
    upper  = strtod(argv[6], 0);       //$y1
    width  = strtol(argv[7], 0, 10);   //$w
    height = strtol(argv[8], 0, 10);   //$h

    /* allocate memory for image */

    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    /* create threads & join threads */

    for (int i=0; i<thread_num; i++){
        tids[i] = i;
        pthread_create(&threads[i], NULL, work, (void*)&tids[i]);
    }
    for (int i=0; i<thread_num; i++) {
        pthread_join(threads[i], NULL);
    }//main thread 等待其他 threads 結束

    /* draw and cleanup */
    
    write_png(filename, iters, width, height, image);
    free(image);
    return 0;
}

/* compile & judge */
//(copy to hw2a.cc)
//compile: "g++ -lpng  -lm -O3 -pthread hw2a.cc -o hw2a" or "make hw2a"
//execute: srun -n1 -c5 ./hw2a out_hw2a_basic.png 10000 -2 2 -2 2 800 800
//judge: hw2a-judge (about 751s)