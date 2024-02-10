#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <nmmintrin.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

int thread_num, rows_per_process, total_row, iters, width, height;
int* image;
double left, right, lower, upper;

void work(int total_processes, int rank, int* image_r){
/* mandelbrot set */

    double range_per_unit_h = (upper-lower)/height;
    double range_per_unit_w = (right-left)/width;
    int index = 0;

	__m128d twos = _mm_set_pd1(2);
	__m128d fours = _mm_set_pd1(4);

    int odd = width%2==1 ? 1 : 0;
	for (int j = height-1-rank; j>=0; j-=total_processes) {
		double b0 = j * range_per_unit_h + lower;
        __m128d b0s = _mm_load1_pd(&b0);
        #pragma omp parallel num_threads(thread_num) 
		{
            #pragma omp for schedule(dynamic, 1)
            for (int i=0; i<width; i+=2) {//width = 11, thread 可選 i = 0 2 4 6 8
                double a0[2] = {left + i*range_per_unit_w, left + (i+1)*range_per_unit_w};
                __m128d a0s = _mm_load_pd(a0);
                
                int pixel_values[2] = {0,0};
                int finish[2] = {0,0};                    
                __m128d as = _mm_set_pd(0, 0);
                __m128d bs = _mm_set_pd(0, 0);
                __m128d ls = _mm_set_pd(0, 0); //length_squareds

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
                    __m128d temp = _mm_add_pd(_mm_sub_pd(_mm_mul_pd(as, as), _mm_mul_pd(bs, bs)), a0s);
                    bs = _mm_add_pd(_mm_mul_pd(_mm_mul_pd(as, bs), twos), b0s); //b' = 2*a*b + b0
                    as = temp; //a' = a*a - b*b + a0
                    ls = _mm_add_pd(_mm_mul_pd(as, as), _mm_mul_pd(bs, bs)); //length_squared |Zk|^2 = a'*a' + b'*b'               
                }

                image_r[index*width+i] = pixel_values[0]; 
                image_r[index*width+i+1] = pixel_values[1]; 
            }
        }
        if (odd) {
            double a0 = left + (width-1) * range_per_unit_w;
            int pixel_value = 0;
            double a = 0, b = 0, length_squared = 0;
            while (pixel_value < iters && length_squared < 4) {
                double temp = a*a - b*b + a0;
                b = 2*a*b + b0;
                a = temp;
                length_squared = a*a + b*b;
                pixel_value++;
            }
            image_r[index*width+(width-1)] = pixel_value;  
        }
        index++; 
	}
	return;
}

void write_png(const char *filename, const int iters, const int width, const int height, const int *buffer){
/* 將一個圖片緩衝區（buffer）寫入一個 PNG 圖片文件。 */

    /* 設定PNG */

    FILE *fp = fopen(filename, "wb");
	png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	png_infop info_ptr = png_create_info_struct(png_ptr);
	png_init_io(png_ptr, fp);
	png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
	png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
	png_write_info(png_ptr, info_ptr);
	png_set_compression_level(png_ptr, 1);

    /* 將一個由整數陣列 buffer 表示的圖像資料，逐行寫入至一個 PNG 文件。 */

	size_t row_size = 3 * width * sizeof(png_byte);
	png_bytep row = (png_bytep)malloc(row_size);
	int row_index = 0;
	for (int k = 0; k < height; ++k){
		memset(row, 0, row_size);
		for (int x = 0; x < width; ++x){
			int p = buffer[row_index * width + x];
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
        row_index += rows_per_process;
        row_index = row_index>=total_row ? row_index%rows_per_process+1 : row_index;
	}
	free(row);
	png_write_end(png_ptr, NULL);
	png_destroy_write_struct(&png_ptr, &info_ptr);
	fclose(fp);
}

int main(int argc, char **argv){
    /* detect how many CPUs are available */

    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set); //取得可用的core數
    printf("%d cpus available\n", CPU_COUNT(&cpu_set));
    int thread_num = CPU_COUNT(&cpu_set);

    /* argument parsing */

    assert(argc == 9);
    const char* filename = argv[1];    //$out
    iters  = strtol(argv[2], 0, 10);   //$iter
    left   = strtod(argv[3], 0);       //$a
    right  = strtod(argv[4], 0);       //$x1
    lower  = strtod(argv[5], 0);       //$x1
    upper  = strtod(argv[6], 0);       //$y1
    width  = strtol(argv[7], 0, 10);   //$w
    height = strtol(argv[8], 0, 10);   //$h

    /* MPI initialization */

    int rank, total_processes;
	MPI_Init(&argc, &argv); // MPI_Init 是 MPI 程式的啟動點，它初始化 MPI 環境主要是去產生一個MPI Communicator
	MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Group main_group, sub_group;
    MPI_Comm sub_comm;
	if (height < total_processes){
        MPI_Comm_group(MPI_COMM_WORLD, &main_group);
        int processes_range[1][3] = {{0, height-1, 1}};
        MPI_Group_range_incl(main_group, 1, processes_range, &sub_group); // 不管有沒有在範圍內，sub_group 都會收到新值（沒在範圍內的 process's sub_group 應該會收到 MPI_COMM_NULL）
        MPI_Comm_create(MPI_COMM_WORLD, sub_group, &sub_comm);
        if (sub_comm == MPI_COMM_NULL) {// 如果process不在sub_comm中，則終止該process
            MPI_Finalize();
            return 0;
        }
        total_processes = height;
	}
    else{ // 因為之後是以 sub_comm 來操作，但 total_elements 不一定小於 total_processes，所以考慮到其他情況，還是要 assign 值給 sub_somm
        sub_comm = MPI_COMM_WORLD;
    }

	/* allocate work to processes & mandelbrot set */

	rows_per_process = ceil((double)height / total_processes);
    total_row = rows_per_process * total_processes;
    int* image = (int *)malloc(total_row * width * sizeof(int));
	int* image_r = (int *)malloc(rows_per_process * width * sizeof(int));
	work(total_processes, rank, image_r);

	/* draw and cleanup */

	MPI_Gather(image_r, rows_per_process * width, MPI_INT, image, rows_per_process * width, MPI_INT, 0, sub_comm);
	if (rank == 0)
		write_png(filename, iters, width, height, image);
	
	free(image_r);
	free(image);
	MPI_Finalize();
	return 0;
}

/* compile & judge */
//(copy this code to hw2b.cc)
//module load mpi/latest
//compile: "mpicxx -lm -O3 -msse4.2 -fopenmp hw2b.cc -lpng -o hw2b" or "make hw2b"
//execute: srun -N1 -n4 -c4 ./hw2b hw2b.png 10000 -2 2 -2 2 800 800
//judge: hw2b-judge (about 340s)