/* MPI-IO 範例程式碼 */

#include <cstdio>
#include <cstdlib>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // process's rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of processes

    char *input_filename = argv[1];
    char *output_filename = argv[2];

    MPI_File input_file, output_file;
    float data[1];

    MPI_File_open(MPI_COMM_WORLD, input_filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file); // return input_file
    MPI_File_read_at(input_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE); // data 就是前面宣告的 float data[1];
    MPI_File_close(&input_file);

    printf("rank %d got float: %f\n", rank, data[0]);

    MPI_File_open(MPI_COMM_WORLD, output_filename, MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    MPI_File_write_at(output_file, sizeof(float) * rank, data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&output_file);

    printf("rank %d write float: %f\n", rank, data[0]);

    MPI_Finalize();
    return 0;
}

/* Compilation and execution */

//load module: module load mpi/latest
//compile: mpicxx -O3 -lm mpiio.cc -o mpiio
//execute: "srun -N1 -n4 ./mpiio /home/pp23/share/hw1/testcases/01.in 01.out" or "mpirun -n 4 ./mpiio /home/pp23/share/hw1/testcases/01.in 01.out"