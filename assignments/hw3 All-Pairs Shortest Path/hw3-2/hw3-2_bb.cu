#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define DEV_NO 0
cudaDeviceProp prop;
//======================

#define bf 32
#define bb 64
const int INF = (1 << 30) - 1;
int *h_dist_matrix = NULL, *d_dist_matrix = NULL;
int v_num, e_num, matrix_s, grid_size;
int ceil(int a, int b) { 
    return (a + b - 1) / b;
}// (a/b) 向上取整

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    size_t result_n = fread(&v_num, sizeof(int), 1, file);
    size_t result_m = fread(&e_num, sizeof(int), 1, file);
    grid_size = ceil(v_num, bb);
    matrix_s = grid_size << 6;

    cudaMallocHost((void**)&h_dist_matrix, matrix_s * matrix_s * sizeof(int));
    for (int i = 0; i < matrix_s; ++i) {
        for (int j = 0; j < matrix_s; ++j) {
            h_dist_matrix[i * matrix_s + j] = (i == j) ? 0 : INF;
        }
    }

    int pair[3];
    for (int i = 0; i < e_num; ++i) {
        size_t result_p = fread(pair, sizeof(int), 3, file);
        h_dist_matrix[pair[0] * matrix_s + pair[1]] = pair[2];
    }
    fclose(file);

    cudaMalloc((void**)&d_dist_matrix, matrix_s * matrix_s * sizeof(int));
    cudaMemcpy(d_dist_matrix, h_dist_matrix, (size_t)sizeof(int) * matrix_s * matrix_s, cudaMemcpyHostToDevice);
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
        fwrite(&h_dist_matrix[i * matrix_s + 0], sizeof(int), v_num, outfile);
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

__global__ void phase2(int *dd_matrix, int r, int matrix_s){
    __shared__ int sb_for_pvt[bb][bb];
    __shared__ int sb_for_pro[bb][bb];

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

    if (blockIdx.x == 0){ //同行那些
        i_in_dd_0 = (blockIdx.y >= r) ? i + ((blockIdx.y+1)<<6) : i + (blockIdx.y<<6);
        i_in_dd_1 = (blockIdx.y >= r) ? i + ((blockIdx.y+1)<<6) + bf : i + (blockIdx.y<<6) + bf;
    }
    else { //blockIdx.x == 1
        j_in_dd_0 = (blockIdx.y >= r) ? j + ((blockIdx.y+1)<<6) : j + (blockIdx.y<<6);
        j_in_dd_1 = (blockIdx.y >= r) ? j + ((blockIdx.y+1)<<6) + bf : j + (blockIdx.y<<6) + bf;
    }

    //process data
    sb_for_pro[   i][   j] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_0];
    sb_for_pro[   i][j+bf] = dd_matrix[i_in_dd_0 * matrix_s + j_in_dd_1];
    sb_for_pro[i+bf][   j] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_0];
    sb_for_pro[i+bf][j+bf] = dd_matrix[i_in_dd_1 * matrix_s + j_in_dd_1];

    if (blockIdx.x == 0){
        __syncthreads();
        #pragma unroll 64
        for (int k = 0; k < bb; ++k){
            sb_for_pro[   i][   j] = min((sb_for_pro[   i][k] + sb_for_pvt[k][   j]), sb_for_pro[   i][   j]);
            sb_for_pro[   i][j+bf] = min((sb_for_pro[   i][k] + sb_for_pvt[k][j+bf]), sb_for_pro[   i][j+bf]);
            sb_for_pro[i+bf][   j] = min((sb_for_pro[i+bf][k] + sb_for_pvt[k][   j]), sb_for_pro[i+bf][   j]);
            sb_for_pro[i+bf][j+bf] = min((sb_for_pro[i+bf][k] + sb_for_pvt[k][j+bf]), sb_for_pro[i+bf][j+bf]);
        }
    }
    else {
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

__global__ void phase3(int *dd_matrix, int r, int matrix_s){
    __shared__ int sb_for_row[bb][bb];
    __shared__ int sb_for_col[bb][bb];
    __shared__ int sb_for_pro[bb][bb];

    int i = threadIdx.y;
    int j = threadIdx.x;

    int pvt_i_in_dd_0 = i + (r<<6);
    int pvt_j_in_dd_0 = j + (r<<6);
    int pvt_i_in_dd_1 = i + (r<<6) + bf;
    int pvt_j_in_dd_1 = j + (r<<6) + bf;

    int i_in_dd_0 = (blockIdx.x >= r) ? i + ((blockIdx.x+1)<<6) : i + (blockIdx.x<<6);
    int j_in_dd_0 = (blockIdx.y >= r) ? j + ((blockIdx.y+1)<<6) : j + (blockIdx.y<<6);
    int i_in_dd_1 = (blockIdx.x >= r) ? i + ((blockIdx.x+1)<<6) + bf : i + (blockIdx.x<<6) + bf;
    int j_in_dd_1 = (blockIdx.y >= r) ? j + ((blockIdx.y+1)<<6) + bf : j + (blockIdx.y<<6) + bf;

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

void FW(){
    dim3 block_dim(bf, bf);
    dim3 grid_dim_2(2, grid_size-1);
    dim3 grid_dim_3(grid_size-1, grid_size-1);

	for (int r=0; r<grid_size; ++r){
		phase1<<<         1, block_dim>>>(d_dist_matrix, r, matrix_s);
		phase2<<<grid_dim_2, block_dim>>>(d_dist_matrix, r, matrix_s);
		phase3<<<grid_dim_3, block_dim>>>(d_dist_matrix, r, matrix_s);
	}

    cudaMemcpy(h_dist_matrix, d_dist_matrix, matrix_s * matrix_s * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist_matrix);
}

int main(int argc, char* argv[]) {
    cudaGetDeviceProperties(&prop, DEV_NO);
    printf("maxThreasPerBlock = %d, sharedMemPerBlock = %d\n", prop.maxThreadsPerBlock, prop.sharedMemPerBlock);
    //======================
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