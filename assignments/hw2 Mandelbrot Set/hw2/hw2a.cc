#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <nmmintrin.h>
#include <math.h>
#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
using namespace std;

int thread_num, rows_per_thread, total_row, iters, width, height;
int* image;
double left, right, lower, upper;

void* work(void* tid){
/* mandelbrot set */

    int* id = (int*) tid;
    double range_per_unit_h = (upper-lower)/height;
    double range_per_unit_w = (right-left)/width;
    int row_index = *id * rows_per_thread;

	__m128d twos = _mm_set_pd1(2);
	__m128d fours = _mm_set_pd1(4);

    for (int j = height-1-*id; j>=0; j-=thread_num) {
        double b0 = lower + j*range_per_unit_h;
        __m128d b0s = _mm_load1_pd(&b0);
        for (int i=0; i<width; i+=2) { //width=11 , 0 2 4 6 8 
            if (i+1 < width) {
                double a0[2] = {left + i*range_per_unit_w, left + (i+1)*range_per_unit_w};
                __m128d a0s = _mm_load_pd(a0);

                int pixel_values[2] = {0,0};
                int finish[2] = {0,0};                
                __m128d as = _mm_set_pd(0, 0);
                __m128d bs = _mm_set_pd(0, 0);
                __m128d ls = _mm_set_pd(0, 0); //length_squareds
                __m128d temp;

                while (!finish[0] || !finish[1]) {//因為可能其中一個因為不滿足條件而提早結束迭代
                    /* 判斷部分 */
                    if (!finish[0]) {
                        if (pixel_values[0]<iters && _mm_comilt_sd(ls, fours))
                            pixel_values[0]++;
                        else 
                            finish[0] = 1;
                    }
                    if (!finish[1]) {
                        __m128d ls_T = _mm_shuffle_pd(ls, ls, 1);
                        if (pixel_values[1]<iters && _mm_comilt_sd(ls_T, fours))
                            pixel_values[1]++;
                        else
                            finish[1] = 1;
                    }
                    /* 計算部分 */                  
                    temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(as, as), _mm_mul_pd(bs, bs)), a0s);
                    bs = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(as, bs), twos), b0s); //b' = 2*a*b + b0
                    as = temp; //a' = a*a - b*b + a0
                    ls = _mm_add_pd(_mm_mul_pd(as, as), _mm_mul_pd(bs, bs)); //length_squared |Zk|^2 = a'*a' + b'*b'               
                }
                image[row_index*width+i] = pixel_values[0]; 
                image[row_index*width+i+1] = pixel_values[1]; 
            }
            else {
                double a0 = left + (i * range_per_unit_w);
                int pixel_value = 0;
                double a = 0, b = 0, length_squared = 0;
                while (pixel_value < iters && length_squared < 4) {
                    double temp = a*a - b*b + a0;
                    b = 2*a*b + b0;
                    a = temp;
                    length_squared = a*a + b*b;
                    pixel_value++;
                }
                image[row_index*width+i] = pixel_value; 
            }
        }
        row_index++;
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
    int row_index = 0;
    for (int y = 0; y < height; ++y) {
    //y 僅代表進行第幾次寫入，實際上從 buffer 中取出資料的起始位置是由 row_index 決定的
    //一圈寫入一列 pixels
        memset(row, 0, row_size);
        for (int x = 0; x < width; ++x) {
            int p = buffer[row_index*width+x];
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
		row_index += rows_per_thread;
        row_index = row_index>=total_row ? row_index%rows_per_thread+1 : row_index;
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

    /* allocate the number of rows for each thread & allocate memory for image */
    
    rows_per_thread = ceil((double)height/thread_num); //每條 thread 最多會負責的 row 數
    total_row = rows_per_thread * thread_num; //會在 write_png() 中作為判斷是否超過圖片高度的依據
    image = (int*)malloc(total_row * width * sizeof(int));
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
//(copy this code to hw2a.cc)
//compile: g++ -lpng  -lm -O3 -pthread -msse4.2 hw2a.cc -o hw2a
//execute: srun -n1 -c4 ./hw2a out_hw2a.png 10000 -2 2 -2 2 800 800
//hw2a-judge (about 447s)