#include <stdio.h>
#include <stdlib.h>
//======================
#include <omp.h>
#include <sched.h>
int n_threads;
//======================
#include <cuda.h>
#include <cuda_runtime.h>
#define DEV_NO 0
cudaDeviceProp prop;
//======================

#define bf 32
#define bb 64
#define dn 2
const int INF = (1 << 30) - 1;
int *h_dist_matrix = NULL;
int *d_dist_matrix[2] = {NULL, NULL};
int v_num, e_num, matrix_size, grid_size;
//__constant__ int dim[6]; //n, m, bf, matrix_s, grid_s, bb;
int ceil(int a, int b) { 
    return (a + b - 1) / b;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    size_t result_n = fread(&v_num, sizeof(int), 1, file);
    size_t result_m = fread(&e_num, sizeof(int), 1, file);
    grid_size = ceil(v_num, bb);
    matrix_size = grid_size << 6;

    cudaMallocHost((void**)&h_dist_matrix, matrix_size * matrix_size * sizeof(int));
    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            h_dist_matrix[i * matrix_size + j] = (i == j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < e_num; ++i) {
        size_t result_p = fread(pair, sizeof(int), 3, file);
        h_dist_matrix[pair[0] * matrix_size + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < v_num; ++i) {
        /*
        for (int j = 0; j < v_num; ++j) {
            if (h_dist_matrix[i, j, matrix_s)] >= INF) 
                h_dist_matrix[i, j, matrix_s)] = INF;
        }
        */
        fwrite(&h_dist_matrix[i * matrix_size + 0], sizeof(int), v_num, outfile);
    }
    fclose(outfile);
    cudaFree(h_dist_matrix);
}

__global__ void phase1(int *dd_matrix, int r, int matrix_s){
    __shared__ int sb_for_pro[bb][bb];

    int i = threadIdx.y;
    int j = threadIdx.x;
    int i_in_dd_0 = i + (r<<6);
    int j_in_dd_0 = j + (r<<6);
    int i_in_dd_1 = i + (r<<6) + bf;
    int j_in_dd_1 = j + (r<<6) + bf;

    sb_for_pro[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0];  
    sb_for_pro[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_pro[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_pro[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1];  
    
    #pragma unroll 64
    for (int k = 0; k < bb; ++k){
        __syncthreads();
        sb_for_pro[   i][   j] = min((sb_for_pro[   i][k] + sb_for_pro[k][   j]), sb_for_pro[   i][   j]); 
        sb_for_pro[   i][j+bf] = min((sb_for_pro[   i][k] + sb_for_pro[k][j+bf]), sb_for_pro[   i][j+bf]);
        sb_for_pro[i+bf][   j] = min((sb_for_pro[i+bf][k] + sb_for_pro[k][   j]), sb_for_pro[i+bf][   j]); 
        sb_for_pro[i+bf][j+bf] = min((sb_for_pro[i+bf][k] + sb_for_pro[k][j+bf]), sb_for_pro[i+bf][j+bf]);
    }

    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0] = sb_for_pro[   i][   j];  
    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1] = sb_for_pro[   i][j+bf];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0] = sb_for_pro[i+bf][   j];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1] = sb_for_pro[i+bf][j+bf];
}

__global__ void phase2(int *dd_matrix, int r, int matrix_s, int grid_p){
    __shared__ int sb_for_pvt[bb][bb];
    __shared__ int sb_for_pro[bb][bb];

	int block_x = grid_p + blockIdx.y; // block_i代表此block實際上是在整張大grid上的第幾列
	int block_y = blockIdx.x;
	if ((block_x != r && block_y != r) || (block_x == r && block_y == r))
		return; 

    int i = threadIdx.y;
    int j = threadIdx.x;
    int i_in_dd_0 = i + (r<<6);
    int j_in_dd_0 = j + (r<<6);
    int i_in_dd_1 = i + (r<<6) + bf;
    int j_in_dd_1 = j + (r<<6) + bf;

    //pivot data
    sb_for_pvt[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0];
    sb_for_pvt[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_pvt[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_pvt[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1];

    if (block_x == r){ //同列那些
        j_in_dd_0 = j + (block_y<<6);
        j_in_dd_1 = j + (block_y<<6) + bf;
    }
    else if (block_y == r){ //同行那些
        i_in_dd_0 = i + (block_x<<6);
        i_in_dd_1 = i + (block_x<<6) + bf;
    }

    //process data
    sb_for_pro[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0];
    sb_for_pro[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_pro[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_pro[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1];

    if (block_y == r){
        __syncthreads();
        #pragma unroll 64
        for (int k = 0; k < bb; ++k){
            sb_for_pro[   i][   j] = min((sb_for_pro[   i][k] + sb_for_pvt[k][   j]), sb_for_pro[   i][   j]);
            sb_for_pro[   i][j+bf] = min((sb_for_pro[   i][k] + sb_for_pvt[k][j+bf]), sb_for_pro[   i][j+bf]);
            sb_for_pro[i+bf][   j] = min((sb_for_pro[i+bf][k] + sb_for_pvt[k][   j]), sb_for_pro[i+bf][   j]);
            sb_for_pro[i+bf][j+bf] = min((sb_for_pro[i+bf][k] + sb_for_pvt[k][j+bf]), sb_for_pro[i+bf][j+bf]);
        }
    }
    else if (block_x == r){
        __syncthreads();
        #pragma unroll 64
        for (int k = 0; k < bb; ++k){
            sb_for_pro[   i][   j] = min((sb_for_pvt[   i][k] + sb_for_pro[k][   j]), sb_for_pro[   i][   j]);
            sb_for_pro[   i][j+bf] = min((sb_for_pvt[   i][k] + sb_for_pro[k][j+bf]), sb_for_pro[   i][j+bf]);
            sb_for_pro[i+bf][   j] = min((sb_for_pvt[i+bf][k] + sb_for_pro[k][   j]), sb_for_pro[i+bf][   j]);
            sb_for_pro[i+bf][j+bf] = min((sb_for_pvt[i+bf][k] + sb_for_pro[k][j+bf]), sb_for_pro[i+bf][j+bf]);
        }
    }

    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0] = sb_for_pro[   i][   j];
    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1] = sb_for_pro[   i][j+bf];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0] = sb_for_pro[i+bf][   j];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1] = sb_for_pro[i+bf][j+bf];
}

__global__ void phase3(int *dd_matrix, int r, int matrix_s, int grid_p){
    __shared__ int sb_for_row[bb][bb];
    __shared__ int sb_for_col[bb][bb];
    __shared__ int sb_for_pro[bb][bb];

	// (blockIdx.x, blockIdx.y) 是此thread所在的block在小grid上的座標

	int block_x = grid_p + blockIdx.y; // block_i代表此block實際上是在整張大grid上的第幾列
	int block_y = blockIdx.x;
	if (block_x == r || block_y == r)
		return; 

    int i = threadIdx.y;
    int j = threadIdx.x;

    int pvt_i_in_dd_0 = i + (r<<6);
    int pvt_j_in_dd_0 = j + (r<<6);
    int pvt_i_in_dd_1 = i + (r<<6) + bf;
    int pvt_j_in_dd_1 = j + (r<<6) + bf;

    int i_in_dd_0 = i + (block_x<<6);
    int j_in_dd_0 = j + (block_y<<6);
    int i_in_dd_1 = i + (block_x<<6) + bf;
    int j_in_dd_1 = j + (block_y<<6) + bf;

    //process data
    sb_for_pro[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0];
    sb_for_pro[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_pro[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_pro[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1];    
    
    //row data
    sb_for_row[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + pvt_j_in_dd_0];
    sb_for_row[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + pvt_j_in_dd_1];
    sb_for_row[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + pvt_j_in_dd_0];
    sb_for_row[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + pvt_j_in_dd_1];  
    
    //column data
    sb_for_col[   i][   j] = dd_matrix[pvt_i_in_dd_0 * matrix_s + j_in_dd_0];
    sb_for_col[   i][j+bf] = dd_matrix[pvt_i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_col[i+bf][   j] = dd_matrix[pvt_i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_col[i+bf][j+bf] = dd_matrix[pvt_i_in_dd_1 * matrix_s + j_in_dd_1];

    __syncthreads();
    #pragma unroll 64
    for (int k = 0; k < bb; ++k){
        sb_for_pro[   i][   j] = min((sb_for_row[   i][k] + sb_for_col[k][   j]), sb_for_pro[   i][   j]);
        sb_for_pro[   i][j+bf] = min((sb_for_row[   i][k] + sb_for_col[k][j+bf]), sb_for_pro[   i][j+bf]);
        sb_for_pro[i+bf][   j] = min((sb_for_row[i+bf][k] + sb_for_col[k][   j]), sb_for_pro[i+bf][   j]);
        sb_for_pro[i+bf][j+bf] = min((sb_for_row[i+bf][k] + sb_for_col[k][j+bf]), sb_for_pro[i+bf][j+bf]);
    }

    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0] = sb_for_pro[   i][   j];
    dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1] = sb_for_pro[   i][j+bf];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0] = sb_for_pro[i+bf][   j];
    dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1] = sb_for_pro[i+bf][j+bf]; 
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    n_threads = CPU_COUNT(&cpu_set);
    //======================
    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    //======================
    input(argv[1]);
    #pragma omp parallel num_threads(dn)
    {
        int dID = omp_get_thread_num();
        cudaSetDevice(dID);
        int grid_avg = grid_size / dn;
        int grid_num = (dID == dn-1) ? grid_avg+(grid_size%dn) : grid_avg;
        int grid_pass = dID * grid_avg;
        int grid_start = (grid_pass * bb) * matrix_size;
        cudaMalloc(&(d_dist_matrix[dID]), (size_t)sizeof(int) * matrix_size * matrix_size);
        cudaMemcpy(&(d_dist_matrix[dID][grid_start]), &(h_dist_matrix[grid_start]), (size_t)sizeof(int) * (grid_num * bb) * matrix_size, cudaMemcpyHostToDevice);
        #pragma omp barrier

        dim3 block_dim(bf, bf);
        dim3 grid_dim_2(grid_size, grid_num);
        dim3 grid_dim_3(grid_size, grid_num);

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        for (int r=0; r<grid_size; ++r){
            int pivot_row_start = r * bb * matrix_size;
            if (grid_pass <= r && r < grid_pass+grid_num){
                cudaDeviceEnablePeerAccess(1 - dID, 0);
                cudaMemcpy(&(d_dist_matrix[dn-dID-1][pivot_row_start]), &(d_dist_matrix[dID][pivot_row_start]), (size_t)sizeof(int) * bb * matrix_size, cudaMemcpyDeviceToDevice);
            }
            #pragma omp barrier
            phase1<<<         1, block_dim>>>(d_dist_matrix[dID], r, matrix_size);
            phase2<<<grid_dim_2, block_dim>>>(d_dist_matrix[dID], r, matrix_size, grid_pass);
            phase3<<<grid_dim_3, block_dim>>>(d_dist_matrix[dID], r, matrix_size, grid_pass);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        printf("Total device execution time for device %d: %f milliseconds\n", dID, milliseconds);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);

        cudaMemcpy(&(h_dist_matrix[grid_start]), &(d_dist_matrix[dID][grid_start]), (size_t)sizeof(int) * (grid_num * bb) * matrix_size, cudaMemcpyDeviceToHost);
        cudaFree(d_dist_matrix[dID]);
        #pragma omp barrier
    }
    output(argv[2]);
    return 0;
}

/* compile & execute */

// (copy this code to hw3-3.cu)
// compile in hades01: "nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -lm -Xcompiler="-fopenmp" -o hw3-3 hw3-3.cu" or "make hw3-3"
// execute in hades01: srun -N1 -n1 --gres=gpu:2 ./hw3-3 /home/pp23/share/hw3-3/cases/p31k1 p31k1.out