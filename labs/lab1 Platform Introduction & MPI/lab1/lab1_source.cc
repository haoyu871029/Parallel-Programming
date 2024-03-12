/* Source code (sequential version)*/

#include <assert.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char** argv) {

	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	
	/* Calculation */

	unsigned long long pixels = 0;
	for (unsigned long long x = 0; x < r; x++) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}

	/* Output */

	printf("%llu\n", (4 * pixels) % k);
}

/* compile & execute */

//(copy this code to lab1.cu)
//Load Intel mpi module: module load mpi/latest
//Compile: "mpicxx -std=c++17 -O3    lab1.cc   -o lab1" or "make"
//Execute: srun time ./lab1 2147483647 2147483647
//(10.txt r=2147483647 k=2147483647 ans=256357661)
//Judge: lab1-judge