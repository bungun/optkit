# CUDA Flags
CUXX=nvcc
CUFLAGS=-arch=sm_50 -Xcompiler '-fPIC'
CULDFLAGS_=-lstdc++ -lcudart -lcublas -lcusparse


# Set CUDA_HOME= path/to/CUDA/libraries/
ifndef CUDAHOME
ifeq ($(shell uname -s), Darwin)
CUDAHOME=/usr/local/cuda/lib/
else
CUDAHOME=/usr/local/cuda/lib64/
endif
endif

# Check System Args.
ifeq ($(shell uname -s), Darwin)
CULDFLAGS=-L$(CUDAHOME) -L/usr/local/lib $(CULDFLAGS_)
SHARED=dylib
else
CULDFLAGS=-L$(CUDAHOME) $(CULDFLAGS_)
SHARED=so
endif

# C++ Flags
CXX=g++
CXXFLAGS=$(IFLAGS) -g -O3 -I. -std=c++11 -Wall -Wconversion -fPIC



.PHONY: default, libcml, cml
default: cml

libcml: link.o
	g++  -shared -Wl,-soname,libtest.so -o $@.$(SHARED) cml_matrix_c.o link.o -L$(CUDAHOME)  -lcudart

cml: cml_matrix_c.o cml_vector_c.o

link.o: cml_matrix_c.cu  cml_matrix_c.cuh
	nvcc  $(CUFLAGS) -dc cml_matrix_c.cu
	nvcc  $(CUFLAGS) -dlink cml_matrix_c.o -o link.o

cml_vector_c.o: cml_vector_c.cu cml_vector_c.cuh 
	$(CUXX) -I. $(CUFLAGS) $< -dc -o $@ $(CULDFLAGS)

cml_matrix_c.o: cml_matrix_c.cu cml_matrix_c.cuh 
	$(CUXX) -I. $(CUFLAGS) $< -dc -o $@ $(CULDFLAGS)

.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ cml_vector cml_matrix 
	rm -rf *.dSYM
