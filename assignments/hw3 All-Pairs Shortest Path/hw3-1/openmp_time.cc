#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

const int INF = ((1 << 30) - 1);
const int V = 50010;
int n, m;
static int Dist[V][V];
double temp, input_time, output_time, compute_time, total_time;

void input(char* infile) {
    temp = omp_get_wtime();
    FILE* file = fopen(infile, "rb");
    fread(&n, sizeof(int), 1, file);
    fread(&m, sizeof(int), 1, file);
    input_time = omp_get_wtime() - temp;

    /* default */
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    /* initialize */
    int pair[3];
    temp = omp_get_wtime();
    for (int i = 0; i < m; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }

    fclose(file);
    input_time += omp_get_wtime() - temp;
    
    return;
} //Dist array ok.

void output(char* outFileName) {
    output_time = omp_get_wtime();
    FILE* outfile = fopen(outFileName, "w");
    for (int i=0; i<n; ++i) {
        for (int j=0; j<n; ++j) {
            if (Dist[i][j] >= INF) 
                Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), n, outfile);
    }
    fclose(outfile);
    output_time = omp_get_wtime() - output_time;
    return;
}

void FW(){
    compute_time = omp_get_wtime();
    for (int k=0; k<n; ++k){
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i=0; i<n; ++i){
            for (int j=0; j<n; ++j){
                if (Dist[i][k] != INF && Dist[i][j] > Dist[i][k] + Dist[k][j])
                    Dist[i][j] = Dist[i][k] + Dist[k][j];
            }
        }
    }
    compute_time = omp_get_wtime() - compute_time;
}

int main(int argc, char* argv[]) {
    total_time = omp_get_wtime();
    input(argv[1]);
    FW();
    output(argv[2]);
    total_time = omp_get_wtime() - total_time;

    printf("\nTotal time:  %f\n", total_time);
	printf("    I/O time: %f\n", input_time + output_time);
	printf("    Compute time: %f\n", compute_time);
    printf("    Other time: %f\n\n", total_time-compute_time-input_time-output_time);
    return 0;
}

//copy to hw3-1.cc
//compile: "g++ -O3 -fopenmp -o hw3-1 hw3-1.cc" or "make"
//execute: srun -N1 -n1 -c5 ./hw3-1 /home/pp23/share/hw3-1/cases/c01.1 c01.1.out