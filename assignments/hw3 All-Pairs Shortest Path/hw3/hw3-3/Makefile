NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
XCFLAGS := -fopenmp
LDFLAGS  := -lm
EXES     := hw3-3

alls: $(EXES)

clean:
	rm -f $(EXES)

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $?
