CC = gcc
CXX = g++
NVCC = nvcc
CXXFLAGS = -O3 -fopenmp
CFLAGS = -O3 -lm -fopenmp
NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61
LDFLAGS = -lm
XCFLAGS = -fopenmp

EXES = hw3-1 hw3-2 hw3-3

.PHONY: all clean

all: $(EXES)

clean:
	rm -f $(EXES)

hw3-1: hw3-1.cc
	$(CXX) $(CXXFLAGS) -o $@ $^

hw3-2: hw3-2.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -o $@ $^

hw3-3: hw3-3.cu
	$(NVCC) $(NVFLAGS) $(LDFLAGS) -Xcompiler="$(XCFLAGS)" -o $@ $^
