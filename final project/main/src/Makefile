CC = gcc
CXX = g++
CXXFLAGS = -O3 -fopenmp -std=c++11
CFLAGS = -O3 -lm
TARGETS = main
OUTPUT = ../bin/main
# add the following include path -I /usr/local/include/boost/
INCLUDE = -I /usr/local/include/boost/

.PHONY: all
# create the hw2 executable to ../bin
# remove the main in current directory
all: $(TARGETS)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $(OUTPUT) $(TARGETS).cpp
	rm -f $(TARGETS)

.PHONY: clean
# remove the hw2 executable in ../bin
clean:
	rm -f $(OUTPUT)