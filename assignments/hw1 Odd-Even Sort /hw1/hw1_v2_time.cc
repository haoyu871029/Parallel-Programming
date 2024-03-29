/* hw1 version 2: 先利用 boundary array 判斷  */

#include <cstdio>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h> 
#include <math.h> 
#include <mpi.h>
#include <boost/sort/spreadsort/float_sort.hpp>
#include <vector>
#include <algorithm>
using namespace std;

vector<float> mergeAndSort(float *a, int size_a, float *b, int size_b) {
    vector<float> merged(size_a + size_b);
    int i = 0, j = 0, k = 0;
    while (i < size_a && j < size_b) {
        if (a[i] < b[j])
            merged[k++] = a[i++];
        else
            merged[k++] = b[j++];
    }
    while (i < size_a) {
        merged[k++] = a[i++];
    }
    while (j < size_b) {
        merged[k++] = b[j++];
    }
    return merged;
}

int main(int argc, char** argv) {
	
    /* MPI initialization & preprocessing */

    int rank, total_processes;
    double computing_time = 0.0;
	double process_start_time, process_end_time;
	MPI_Init(&argc, &argv);
    process_start_time = MPI_Wtime();
    MPI_Comm_size(MPI_COMM_WORLD, &total_processes);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int total_elements = atoi(argv[1]);
    MPI_Group main_group, sub_group;
    MPI_Comm sub_comm;
    if (total_elements < total_processes){
        MPI_Comm_group(MPI_COMM_WORLD, &main_group);
        int processes_range[1][3] = {{0, total_elements-1, 1}};
        MPI_Group_range_incl(main_group, 1, processes_range, &sub_group);
        MPI_Comm_create(MPI_COMM_WORLD, sub_group, &sub_comm);
        if (sub_comm == MPI_COMM_NULL) {
            MPI_Finalize();
            return 0;
        }
        total_processes = total_elements;
    }
    else{
        sub_comm = MPI_COMM_WORLD;
    }
    int last_process_rank = total_processes - 1;

    /* Process settings */
    
    int basic_num = total_elements / total_processes; 
    int extra_num = total_elements % total_processes;

    int actual_num, index_start;
    if (rank < extra_num){
        index_start = rank * (basic_num+1);
        actual_num = basic_num + 1;
    }
    else{
        index_start = (rank*basic_num) + extra_num;
        actual_num = basic_num;
    }

    float *data = new float[actual_num];
    float *recv = new float[basic_num+1];

    int recv_num_from_befo, recv_num_from_next;
    if (rank+1 == extra_num)
        recv_num_from_next = actual_num-1;
    else
        recv_num_from_next = actual_num;   
    if (rank-1 == extra_num-1)
        recv_num_from_befo = actual_num+1;
    else
        recv_num_from_befo = actual_num;

    /* Read input file */

    double io_time = 0.0;
    double io_start_time, io_end_time;
    MPI_File input_file;
    MPI_File_open(sub_comm, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    io_start_time = MPI_Wtime();
    MPI_File_read_at(input_file, sizeof(float)*index_start, data, actual_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    io_end_time = MPI_Wtime();
    io_time += (io_end_time - io_start_time);
    MPI_File_close(&input_file);
    boost::sort::spreadsort::float_sort(data, data + actual_num);

    /* Odd-even sort */

    vector<float> merged;
    int local_swap = 1;
    int global_swap = 1;
    bool data_changed = false;
    float boundary_data[2];
    double comm_time = 0.0;
    double comm_start_time, comm_end_time;

    while (global_swap!=0){
        local_swap = 0;
        global_swap = 0;
        data_changed = false;

        /* even phase: (even,odd) */
        if (!(rank&1) && rank!=last_process_rank){
            boundary_data[0] = data[actual_num-1];
            comm_start_time = MPI_Wtime();
            MPI_Sendrecv(&boundary_data[0], 1, MPI_FLOAT, rank+1, 0,
                         &boundary_data[1], 1, MPI_FLOAT, rank+1, 0,
                         sub_comm, MPI_STATUS_IGNORE);
            comm_end_time = MPI_Wtime();
            comm_time += (comm_end_time-comm_start_time);
            if(boundary_data[0] > boundary_data[1]) {
                data_changed = true;
                comm_start_time = MPI_Wtime();
                MPI_Sendrecv(data, actual_num, MPI_FLOAT, rank+1, 0,
                             recv, recv_num_from_next, MPI_FLOAT, rank+1, 0,
                             sub_comm, MPI_STATUS_IGNORE);
                comm_end_time = MPI_Wtime();
                comm_time += (comm_end_time-comm_start_time);
                merged = mergeAndSort(data, actual_num, recv, recv_num_from_next);
                copy(merged.begin(), merged.begin()+actual_num, data);
            }
        }
        else if (rank&1){
            boundary_data[0] = data[0];
            comm_start_time = MPI_Wtime();
            MPI_Sendrecv(&boundary_data[0], 1, MPI_FLOAT, rank-1, 0,
                         &boundary_data[1], 1, MPI_FLOAT, rank-1, 0,
                         sub_comm, MPI_STATUS_IGNORE);
            comm_end_time = MPI_Wtime();
            comm_time += (comm_end_time-comm_start_time);
            if(boundary_data[0] < boundary_data[1]){
                data_changed = true;
                comm_start_time = MPI_Wtime();
                MPI_Sendrecv(data, actual_num, MPI_FLOAT, rank-1, 0,
                             recv, recv_num_from_befo, MPI_FLOAT, rank-1, 0,
                             sub_comm, MPI_STATUS_IGNORE);      
                comm_end_time = MPI_Wtime();
                comm_time += (comm_end_time-comm_start_time);     
                merged = mergeAndSort(data, actual_num, recv, recv_num_from_befo); 
                copy(merged.begin()+recv_num_from_befo, merged.end(), data); //copy function is supported by <algorithm>
            }
        }

        /* odd phase: (odd,even) */
        if (rank&1 && rank!=last_process_rank){
            boundary_data[0] = data[actual_num-1];
            comm_start_time = MPI_Wtime();
            MPI_Sendrecv(&boundary_data[0], 1, MPI_FLOAT, rank+1, 0,
                         &boundary_data[1], 1, MPI_FLOAT, rank+1, 0,
                         sub_comm, MPI_STATUS_IGNORE);
            comm_end_time = MPI_Wtime();
            comm_time += (comm_end_time-comm_start_time);
            if(boundary_data[0] > boundary_data[1]) {
                data_changed = true;
                comm_start_time = MPI_Wtime();
                MPI_Sendrecv(data, actual_num, MPI_FLOAT, rank+1, 0,
                             recv, recv_num_from_next, MPI_FLOAT, rank+1, 0,
                             sub_comm, MPI_STATUS_IGNORE);
                comm_end_time = MPI_Wtime();
                comm_time += (comm_end_time-comm_start_time);
                merged = mergeAndSort(data, actual_num, recv, recv_num_from_next);
                copy(merged.begin(), merged.begin()+actual_num, data);
            }
        }
        else if (!(rank&1) && rank!=0){
            boundary_data[0] = data[0];
            comm_start_time = MPI_Wtime();
            MPI_Sendrecv(&boundary_data[0], 1, MPI_FLOAT, rank-1, 0,
                         &boundary_data[1], 1, MPI_FLOAT, rank-1, 0,
                         sub_comm, MPI_STATUS_IGNORE);
            comm_end_time = MPI_Wtime();
            comm_time += (comm_end_time-comm_start_time);
            if(boundary_data[0] < boundary_data[1]){
                data_changed = true;
                comm_start_time = MPI_Wtime();
                MPI_Sendrecv(data, actual_num, MPI_FLOAT, rank-1, 0,
                            recv, recv_num_from_befo, MPI_FLOAT, rank-1, 0,
                            sub_comm, MPI_STATUS_IGNORE);      
                comm_end_time = MPI_Wtime();
                comm_time += (comm_end_time-comm_start_time);
                merged = mergeAndSort(data, actual_num, recv, recv_num_from_befo); 
                copy(merged.begin()+recv_num_from_befo, merged.end(), data);
            }
        } 

        /* check any swap */
        local_swap = data_changed ? 1 : 0;
        MPI_Allreduce(&local_swap, &global_swap, 1, MPI_INT, MPI_LOR, sub_comm);
    }

    /* Write output file */
    
    MPI_File output_file;
    MPI_File_open(sub_comm, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
    io_start_time = MPI_Wtime();
    MPI_File_write_at(output_file, sizeof(float) * index_start, data, actual_num, MPI_FLOAT, MPI_STATUS_IGNORE);
    io_end_time = MPI_Wtime();
    io_time += (io_end_time - io_start_time);
    MPI_File_close(&output_file);

    process_end_time = MPI_Wtime();
    computing_time += (process_end_time-process_start_time)-(comm_time+io_time);

    /* calculate average communication time */

    double avg_comm_time = 0.0;
    MPI_Reduce(&comm_time, &avg_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
    if (rank == 0) {
        avg_comm_time = avg_comm_time/total_processes;
    }

    /* calculate average IO time */

    double avg_io_time = 0.0;
    MPI_Reduce(&io_time, &avg_io_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
    if (rank == 0) {
        avg_io_time = avg_io_time/total_processes;
    }

    /* calculate average computing time */

    double avg_computing_time = 0.0;
    MPI_Reduce(&computing_time, &avg_computing_time, 1, MPI_DOUBLE, MPI_SUM, 0, sub_comm);
    if (rank == 0) {
        avg_computing_time = avg_computing_time/total_processes;
    }

    /* output time profile */
    
    if (rank == 0) {
        printf("-----------------------------------\n");
        printf("avg_comm_time = %lf\n", avg_comm_time);
        printf("avg_io_time = %lf\n", avg_io_time);
        printf("avg_computing_time = %lf\n", avg_computing_time);
        printf("-----------------------------------\n");
        printf("run_time = %lf\n", process_end_time-process_start_time);
        printf("total_iteration = %lf\n", iteration);
        printf("-----------------------------------\n");
    }

    MPI_Finalize();
    return 0;
}

/* Compilation and execution */

//copy this code to hw1.cc
//load module: module load mpi/latest
//compile: "mpicxx -O3 -lm hw1.cc -o hw1" or "make"
//execute: srun -N1 -n5 ./hw1 4 01.in 01.out
//check: hw1-floats /home/pp23/share/hw1/testcases/01.out 01.out
//judge: hw1-judge