/* Implementation */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);

    /* MPI Initialization */

	int rc = MPI_Init(&argc, &argv);
	if (rc != MPI_SUCCESS){
		printf("Error");
		MPI_Abort(MPI_COMM_WORLD, rc); //MPI_Abort 將被調用來終止 MPI 環境，會中止所有在一個通訊器（在這個情況下是 MPI_COMM_WORLD）內的 MPI 過程。
	}

	int rank, size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 計算每個 process 負責的 x 範圍（0~r-1中的幾到幾）*/

    unsigned long long local_start = rank * (r / size);
    unsigned long long local_end = (rank + 1) * (r / size);
    if(rank == size-1) {
        local_end = r;
    }

    /* Calculation */

    unsigned long long local_pixels = 0;
    for (unsigned long long x = local_start; x < local_end; x++) {
        unsigned long long y = ceil(sqrtl(r*r - x*x));
        local_pixels += y;
        local_pixels %= k;
    }

    unsigned long long global_pixels = 0;
    MPI_Reduce(&local_pixels, &global_pixels, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD); //加總所有 processes 的 local_pixels，將值存到 rank0 process 的 global_pixels

    /* Output */

    if (rank == 0) {
        printf("%llu\n", (4 * global_pixels) % k);
    }
    /*
    if (rank == 0) {
        printf("rank %d of %d, local_pixels = %llu, total pixels %k = %llu\n", rank, size, local_pixels, (4 * global_pixels) % k);
    }
    else {
        printf("rank %d of %d, local_pixels = %llu\n", rank, size, local_pixels);
    }
    */

	MPI_Finalize();
	return 0;
}

/* compile & execute */

//(copy this code to lab1.cu)
//Load Intel mpi module: module load mpi/latest
//Compile: "mpicxx -std=c++17 -O3    lab1.cc   -o lab1" or "make"
//Execute: srun -n10 time ./lab1 2147483647 2147483647
//(10.txt r=2147483647 k=2147483647 ans=256357661)
//Judge: lab1-judge