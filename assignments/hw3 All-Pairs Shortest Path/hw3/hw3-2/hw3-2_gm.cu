#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
cudaDeviceProp prop;
//=====================

#define bf 32
const int INF = (1 << 30) - 1;
int* h_dist_matrix;
int* d_dist_matrix;
int v_num, e_num, matrix_size, grid_size;
int ceil(int a, int b) { 
    return (a + b - 1) / b;
}// (a/b) 向上取整
__device__ __host__ size_t _2d_to_1d(int i, int j, int row_size){
	return i * row_size + j;
}

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    size_t result_n = fread(&v_num, sizeof(int), 1, file);
    size_t result_m = fread(&e_num, sizeof(int), 1, file);
    grid_size = ceil(v_num, bf);
    matrix_size = grid_size <<5;
    h_dist_matrix = (int*)malloc(matrix_size * matrix_size * sizeof(int));

    for (int i = 0; i < matrix_size; ++i) {
        for (int j = 0; j < matrix_size; ++j) {
            h_dist_matrix[i * matrix_size + j] = (i == j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < e_num; ++i) {
        size_t result_p = fread(pair, sizeof(int), 3, file);
        h_dist_matrix[_2d_to_1d(pair[0], pair[1], matrix_size)] = pair[2];
    }
    fclose(file);

    cudaMalloc((void**)&d_dist_matrix, matrix_size * matrix_size * sizeof(int));
    cudaMemcpy(d_dist_matrix, h_dist_matrix, matrix_size * matrix_size * sizeof(int), cudaMemcpyHostToDevice);
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
    free(h_dist_matrix);
    h_dist_matrix = NULL;
}

__global__ void phase1(int *dd_matrix, int r, int matrix_s){
    int i = threadIdx.y;
    int j = threadIdx.x;
    int offset = r<<5;
    
    #pragma unroll 32
    for (int k = offset; k < (r+1)<<5; ++k){
        __syncthreads();
        dd_matrix[_2d_to_1d(i+offset, j+offset, matrix_s)] = min((dd_matrix[_2d_to_1d(i+offset, k, matrix_s)] + dd_matrix[_2d_to_1d(k, j+offset, matrix_s)])
                                                                , dd_matrix[_2d_to_1d(i+offset, j+offset, matrix_s)]);
    }
}

__global__ void phase2(int *dd_matrix, int r, int matrix_s){
    int i = threadIdx.y;
    int j = threadIdx.x;
    int i_offset, j_offset;
    int offset = r<<5;

    if (blockIdx.x == 0){//同行的
        i_offset = (blockIdx.y >= r) ? (blockIdx.y+1)<<5 : blockIdx.y<<5;
        j_offset = offset; 
    }
    else{ // blockIdx.x != 0 同列的
        i_offset = offset;
        j_offset = (blockIdx.y >= r) ? (blockIdx.y+1)<<5 : blockIdx.y<<5;
    }
    
    __syncthreads();
    #pragma unroll 32
    for (int k = offset; k < (r+1)<<5; ++k){
        dd_matrix[_2d_to_1d(i+i_offset, j+j_offset, matrix_s)] = min((dd_matrix[_2d_to_1d(i+i_offset, k, matrix_s)] + dd_matrix[_2d_to_1d(k, j+j_offset, matrix_s)])
                                                                    ,dd_matrix[_2d_to_1d(i+i_offset, j+j_offset, matrix_s)]);
    }
}

__global__ void phase3(int *dd_matrix, int r, int matrix_s){
    int i = threadIdx.y;
    int j = threadIdx.x;
    int offset = r<<5;

    int i_offset = (blockIdx.x >= r) ? (blockIdx.x+1)<<5 : blockIdx.x<<5;
    int j_offset = (blockIdx.y >= r) ? (blockIdx.y+1)<<5 : blockIdx.y<<5;

    __syncthreads();
    #pragma unroll 32
    for (int k = offset; k < (r+1)<<5; ++k){
        dd_matrix[_2d_to_1d(i+i_offset, j+j_offset, matrix_s)] = min((dd_matrix[_2d_to_1d(i+i_offset, k, matrix_s)] + dd_matrix[_2d_to_1d(k, j+j_offset, matrix_s)])
                                                                    ,dd_matrix[_2d_to_1d(i+i_offset, j+j_offset, matrix_s)]);
    }
}

void FW(){
    dim3 block_dim(bf, bf);
    dim3 grid_dim_2(2, grid_size-1);
    dim3 grid_dim_3(grid_size-1, grid_size-1);

	for (int r=0; r<grid_size; ++r){
		phase1<<<         1, block_dim>>>(d_dist_matrix, r, matrix_size);
		phase2<<<grid_dim_2, block_dim>>>(d_dist_matrix, r, matrix_size);
		phase3<<<grid_dim_3, block_dim>>>(d_dist_matrix, r, matrix_size);
	}
    cudaMemcpy(h_dist_matrix, d_dist_matrix, matrix_size * matrix_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist_matrix);
}

int main(int argc, char* argv[]) {
    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    //=====================
    input(argv[1]);
    FW();
    output(argv[2]);
    return 0;
}

/* compile & execute */

// (copy this code to hw3-2.cu)
// compile in hades01: "nvcc -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -lm -o hw3-2 hw3-2.cu" or "make hw3-2"
// execute in hades01: srun -N1 -n1 --gres=gpu:1 ./hw3-2 /home/pp23/share/hw3-2/cases/p20k1 p20k1.out
// judge in hades01: hw3-2-judge