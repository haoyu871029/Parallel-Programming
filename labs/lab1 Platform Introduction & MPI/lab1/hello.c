#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //the rank (id) of the calling process
    MPI_Comm_size(MPI_COMM_WORLD, &size); //the total number of process

    printf("Hello, World.  I am %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}

/* Compile and run the program */
//Load Intel mpi module: module load mpi/latest
//Compile: mpicc -O3 hello.c -o hello
//Execute: srun -N2 -n24 ./hello (2 nodes 代表有 24 cores 可用，因此在每個 process 用一個 cpu 的情況下，可以 launch 24 processes)