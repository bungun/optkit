# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
INCLUDE=$(OPTKITROOT)include/
PREFIX_OUT=$(OUT)optkit_

LINSYS=linsys/optkit_
LASRC=$(SRC)$(LINSYS)

OPERATOR=operator/optkit_operator_
OPSRC=$(SRC)$(OPERATOR)

# C Flags
CC=gcc
CCFLAGS= -g -O3 -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external 
CCFLAGS+=-I$(INCLUDE)linsys -I$(INCLUDE)operator
CCFLAGS+=-Wall -Wconversion -Wpedantic -Wno-unused-function -std=c99
LDFLAGS_=-lstdc++ -lm

# C++ Flags
CXX=g++
CXXFLAGS= -g -O3 -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external
CXXFLAGS+=-I$(INCLUDE)linsys -I$(INCLUDE)operator
CXXFLAGS+=-Wall -Wconversion -Wpedantic -Wno-unused-function

# CUDA Flags
CUXX=nvcc
CUXXFLAGS=-arch=sm_50 -Xcompiler -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external
CUXXFLAGS+=-I$(INCLUDE)linsys -I$(INCLUDE)operator
CULDFLAGS_=-lstdc++ -lm

# Darwin / Linux
ifeq ($(shell uname -s), Darwin)
LDFLAGS_+=-framework Accelerate
CULDFLAGS_+=-L/usr/local/cuda/lib 
SHARED=dylib
ifdef USE_OPENMP
CCFLAGS+=-fopenmp=libopenmp
CXXFLAGS+=-fopenmp=libopenmp
endif
else
LDFLAGS_+=-lblas
CULDFLAGS_+=-L/usr/local/cuda/lib64 
SHARED=so
ifdef USE_OPENMP
CCFLAGS+=-fopenmp
CXXFLAGS+=-fopenmp
endif
endif

CULDFLAGS_+=-lcudart -lcublas -lcusparse

# make options
ifndef GPU
GPU=0
endif

ifndef FLOAT
FLOAT=0
endif

# optional compiler flags 
OPT_FLAGS=
ifneq ($(FLOAT), 0)
OPT_FLAGS+=-DFLOAT # use floats rather than doubles
endif
ifdef OPTKIT_DEBUG_PYTHON
OPT_FLAGS+=-DOK_DEBUG_PYTHON
endif

CCFLAGS+=$(OPT_FLAGS)
CXXFLAGS+=$(OPT_FLAGS)
CUXXFLAGS+=$(OPT_FLAGS)

# make switches
ifneq ($(FLOAT), 0)
PRECISION=32
else
PRECISION=64
endif

ifneq ($(GPU), 0)
LDFLAGS=$(CULDFLAGS_)
DEVICETAG=gpu
else
LDFLAGS=$(LDFLAGS_)
DEVICETAG=cpu
endif

LIBCONFIG=$(DEVICETAG)$(PRECISION)

VECTOR_TARG=$(DEVICETAG)_vector
DENSE_TARG=$(DEVICETAG)_dense
SPARSE_TARG=$(DEVICETAG)_sparse
PROX_TARG=$(DEVICETAG)_prox

LINSYSLIBS=liblinsys_dense liblinsys_sparse
POGSLIBS=libpogs_dense

VECTOR_SRC=$(LASRC)vector.cpp
VECTOR_OBJ=$(OUT)$(LINSYS)vector_$(LIBCONFIG).o
DENSE_CPU_SRC=$(LASRC)vector.cpp $(LASRC)matrix.cpp $(LASRC)blas.c 
DENSE_CPU_SRC+=$(LASRC)dense.c
DENSE_GPU_SRC=$(LASRC)matrix.cu $(LASRC)matrix.cu $(LASRC)blas.cu 
DENSE_GPU_SRC+=$(LASRC)dense.cu
DENSE_OBJ=$(patsubst $(LASRC)%.cu,$(OUT)%$(LIBCONFIG).o,$(DENSE_GPU_SRC))
SPARSE_OBJ=$(LASRC)sparse_$(LIBCONFIG).o


PROX_OBJ=$(PREFIX_OUT)$(PROXTARG)$(PRECISION).o

OPERATOR_SRC=$(OPSRC)dense.c $(OPSRC)sparse.c $(OPSRC)diagonal.c 
OPERATOR_OBJ=$(patsubst $(OPSRC)%.c,$(OUT)%.o,$(OPERATOR_SRC))

CG_OBJ=$(PREFIX_OUT)cg_$(LIBCONFIG).o
PROJ_OBJ=$(PREFIX_OUT)projector_$(LIBCONFIG).o
PROJ_DIRECT_OBJ=$(PREFIX_OUT)projector_direct_$(LIBCONFIG).o
EQUIL_OBJ=$(PREFIX_OUT)equil_$(LIBCONFIG).o
EQUIL_DENSE_OBJ=$(PREFIX_OUT)equil_dense_$(LIBCONFIG).o
POGS_OBJ=$(PREFIX_OUT)pogs_$(LIBCONFIG).o
POGS_ABSTRACT_OBJ=$(PREFIX_OUT)pogs_abstract_$(LIBCONFIG).o

SPARSE_STATIC_DEPS=$(VECTOR_OBJ)
CG_STATIC_DEPS=$(DENSE_OBJ) $(CG_OBJ)
OPERATOR_STATIC_DEPS=$(DENSE_OBJ) $(SPARSE_OBJ) $(OPERATOR_OBJ)
EQUIL_STATIC_DEPS=$(DENSE_OBJ) $(SPARSE_OBJ) $(OPERATOR_OBJ) $(EQUIL_OBJ) 
PROJ_STATIC_DEPS=$(DENSE_OBJ) $(CG_OBJ) $(PROJ_OBJ)
POGS_STATIC_DEPS=$(POGS_OBJ) $(EQUIL_OBJ) $(PROJ_OBJ)
POGS_STATIC_DEPS+=$(DENSE_OBJ) $(PROX_OBJ)
POGS_ABSTRACT_STATIC_DEPS=$(POGS_ABSTRACT_OBJ) $(EQUIL_OBJ) $(PROJ_OBJ)
POGS_ABSTRACT_STATIC_DEPS+=$(DENSE_OBJ) $(SPARSE_OBJ) $(PROX_OBJ) 
POGS_ABSTRACT_STATIC_DEPS+=$(CG_OBJ) $(OPERATOR_OBJ)

.PHONY: default, all, libs, libok, libok_dense, libok_sparse, libprox
.PHONY: libpogs
.PHONY: libequil, equil, libprojector, projector, operator
default: cpu_dense
all: libs libequil libprojector libpogs
libs: libok libprox
libok: liblinsys_dense liblinsys_sparse
libok_sparse: liblinsys_sparse
libok_dense: liblinsys_dense
libpogs: libpogs_dense

libpogs_abstract: pogs_abstract equil projector $(LINSYSLIBS) libprox
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED)  \
	$(POGS_ABSTRACT_STATIC_DEPS) $(LDFLAGS) 

libpogs_sparse: pogs equil projector $(LINSYSLIBS) libprox
	mkdir -p $(OUT) \	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED)  \
	$(POGS_STATIC_DEPS) $(SPARSE_OBJ) $(LDFLAGS)

libpogs_dense: pogs equil_dense projector_direct $(LINSYSLIBS) libprox
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED)  \
	$(POGS_STATIC_DEPS) $(LDFLAGS) 

libprojector: projector cg $(DENSE_TARG) $(SPARSE_TARG) 
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED) \
	$(PROJ_STATIC_DEPS) $(LDFLAGS)

libequil: equil $(DENSE_TARG) $(SPARSE_TARG) operator
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED)  \
	$(EQUIL_STATIC_DEPS) $(LDFLAGS)

libcg: cg $(VECTOR_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o\
	$(OUT)$@_$(LIBCONFIG).$(SHARED)  \
	$(CG_STATIC_DEPS) $(LDFLAGS) 

liboperator: operator $(DENSE_TARG) $(SPARSE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED) \
	$(OPERATOR_STATIC_DEPS) $(LDFLAGS)

libprox: $(PROX_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED) \
	$(PROX_OBJ) $(LDFLAGS)

liblinsys_sparse: $(SPARSE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED) \
	$(SPARSE_OBJ) $(SPARSE_STATIC_DEPS) $(LDFLAGS) 

liblinsys_dense: $(DENSE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(LIBCONFIG).$(SHARED) \
	$(DENSE_OBJ) $(LDFLAGS)

pogs_abstract: $(SRC)optkit_abstract_pogs.c
	$(CC) $(CCFLAGS) $< -c -o $(POGS_ABSTRACT_OBJ) \

pogs: $(SRC)optkit_pogs.c
	mkdir -p $(OUT) 
	$(CC) $(CCFLAGS) $< -c -o $(POGS_OBJ)
	
equil: $(SRC)optkit_equilibration.c 
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(EQUIL_OBJ) \

equil_dense: $(SRC)optkit_equilibration.c
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(EQUIL_DENSE_OBJ) -DOPTKIT_NO_OPERATOR_EQUIL

projector: $(SRC)optkit_projector.c
	mkdir -p $(OUT)
	$(CC) $(CXXFLAGS) $< -c -o $(PROJ_OBJ)

projector_direct: $(SRC)optkit_projector.c
	mkdir -p $(OUT)
	$(CC) $(CXXFLAGS) $< -c -o $(PROJ_DIRECT_OBJ) -DOPTKIT_NO_INDIRECT_PROJECTOR

operator: $(OPERATOR_SRC) 
	mkdir -p $(OUT)
	mkdir -p $(OUT)/operator
	$(CC) $(CCFLAGS) $(OPSRC)dense.c  -c -o $(OUT)$(OPERATOR)dense.o
	$(CC) $(CCFLAGS) $(OPSRC)sparse.c  -c -o $(OUT)$(OPERATOR)sparse.o
	$(CC) $(CCFLAGS) $(OPSRC)diagonal.c -c -o $(OUT)$(OPERATOR)diagonal.o

cg: $(SRC)optkit_cg.c
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(CG_OBJ)

cpu_prox: $(SRC)optkit_prox.cpp
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PROX_OBJ)

gpu_prox: $(SRC)optkit_prox.cu
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PROX_OBJ)

cpu_sparse: $(SRC)optkit_sparse.cpp cpu_vector
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CXX) $(CXXFLAGS) $< -c -o $(SPARSE_OBJ)	

gpu_sparse: $(SRC)optkit_sparse.cu gpu_vector
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $(SPARSE_OBJ)

cpu_dense: $(DENSE_CPU_SRC)
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CXX) $(CXXFLAGS) $(LASRC)vector.cpp -c -o \
	$(OUT)$(LINSYS)vector_$(LIBCONFIG).o
	$(CXX) $(CXXFLAGS) $(LASRC)matrix.cpp -c -o \
	$(OUT)$(LINSYS)matrix_$(LIBCONFIG).o
	$(CC) $(CCFLAGS) $(LASRC)blas.c -c -o $(OUT)$(LINSYS)blas_$(LIBCONFIG).o
	$(CC) $(CCFLAGS) $(LASRC)dense.c -c -o $(OUT)$(LINSYS)dense_$(LIBCONFIG).o

gpu_dense: $(DENSE_GPU_SRC)
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CUXX) $(CUXXFLAGS) $(LASRC)vector.cu -c -o \
	$(OUT)$(LINSYS)vector_$(LIBCONFIG).o
	$(CUXX) $(CUXXFLAGS) $(LASRC)matrix.cu -c -o \
	$(OUT)$(LINSYS)matrix_$(LIBCONFIG).o
	$(CUXX) $(CUXXFLAGS) $(LASRC)blas.cu -c -o \
	$(OUT)$(LINSYS)blas_$(LIBCONFIG).o
	$(CUXX) $(CUXXFLAGS) $(LASRC)dense.cu -c -o \
	$(OUT)$(LINSYS)dense_$(LIBCONFIG).o

cpu_vector: $(VECTOR_SRC)
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CXX) $(CXXFLAGS) $< -c -o $(VECTOR_OBJ)

gpu_vector: $(VECTOR_SRC)
	mkdir -p $(OUT)
	mkdir -p $(OUT)/linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $(VECTOR_OBJ)

.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)