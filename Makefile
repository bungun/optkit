# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
INCLUDE=$(OPTKITROOT)include/
PREFIX_OUT=$(OUT)optkit_
OPERATOR=operator/optkit_operator_
OPSRC=$(SRC)$(OPERATOR)

# C Flags
CC=gcc
CCFLAGS= -g -O3 -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external -I$(INCLUDE)operator
CCFLAGS += -Wall -Wconversion -Wpedantic -Wno-unused-function -std=c99
LDFLAGS_=-lstdc++ -lm

# C++ Flags
CXX=g++
CXXFLAGS= -g -O3 -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external -I$(INCLUDE)operator
CXXFLAGS += -Wall -Wconversion -Wpedantic -Wno-unused-function

# CUDA Flags
CUXX=nvcc
CUXXFLAGS=-arch=sm_50 -Xcompiler -fPIC -I. -I$(INCLUDE) -I$(INCLUDE)external -I$(INCLUDE)operator
CULDFLAGS_=-lstdc++ -lm

# Check system args
ifeq ($(shell uname -s), Darwin)
LDFLAGS_ += -framework Accelerate
CULDFLAGS_ += -L/usr/local/cuda/lib 
SHARED=dylib
ifdef USE_OPENMP
CCFLAGS += -fopenmp=libopenmp
CXXFLAGS += -fopenmp=libopenmp
endif
else
LDFLAGS_ += -lblas
CULDFLAGS_ += -L/usr/local/cuda/lib64 
SHARED=so
ifdef USE_OPENMP
CCFLAGS += -fopenmp
CXXFLAGS += -fopenmp
endif
endif

CULDFLAGS_ += -lcudart -lcublas -lcusparse

# make options
ifndef SPARSE
SPARSE = 0
endif

ifndef GPU
GPU = 0
endif

# compiler options
ifndef FLOAT
FLOAT = 0
endif

ifndef ROWMAJOR
ROWMAJOR = 0
endif

ifndef COLMAJOR
COLMAJOR = 0
endif

# optional compiler flags 
OPT_FLAGS=
ifneq ($(FLOAT), 0)
OPT_FLAGS += -DFLOAT # use floats rather than doubles
endif
ifdef OPTKIT_DEBUG_PYTHON
OPT_FLAGS += -DOK_DEBUG_PYTHON
endif

CCFLAGS += $(OPT_FLAGS)
CXXFLAGS += $(OPT_FLAGS)
CUXXFLAGS += $(OPT_FLAGS)

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

DENSETARG=$(DEVICETAG)_dense
SPARSETARG=$(DEVICETAG)_sparse

LINSYSLIBS=libok_dense
POGSLIBS=libpogs_dense
ifneq ($(SPARSE), 0)
LINSYSLIBS += libok_sparse
POGSLIBS += libpogs_sparse
endif
PROXTARG=$(DEVICETAG)_prox

DENSESTATIC=$(PREFIX_OUT)dense_$(ORDER)$(DEVICETAG)$(PRECISION).o
SPARSESTATIC=$(PREFIX_OUT)sparse_$(ORDER)$(DEVICETAG)$(PRECISION).o
PROXSTATIC=$(PREFIX_OUT)$(PROXTARG)$(PRECISION).o
PROJSTATIC=$(PREFIX_OUT)projector_$(DEVICETAG)$(PRECISION).o
EQUILSTATIC=$(PREFIX_OUT)equil_$(DEVICETAG)$(PRECISION).o
POGSSTATIC=$(PREFIX_OUT)pogs_$(DEVICETAG)$(PRECISION).o
POGSABSTRACTSTATIC=$(PREFIX_OUT)pogs_abstract_$(DEVICETAG)$(PRECISION).o

# OPSTATIC=$(PREFIX_OUT)operator_$(DEVICETAG)$(PRECISION).o
OPSTATIC=
CGSTATIC=$(PREFIX_OUT)cg_$(DEVICETAG)$(PRECISION).o

OPERATOR_SRC=$(OPSRC)dense.c $(OPSRC)sparse.c $(OPSRC)diagonal.c 
OPERATOR_OBJ=$(patsubst $(SRC)%.c,$(OUT)%.o,$(OPERATOR_SRC))

SPARSE_STATIC_DEPS=$(DENSESTATIC)
CG_STATIC_DEPS=$(DENSESTATIC) $(CGSTATIC)
OPERATOR_STATIC_DEPS=$(DENSESTATIC) $(SPARSESTATIC) $(OPSTATIC)
EQUIL_STATIC_DEPS=$(DENSESTATIC) $(SPARSESTATIC) $(OPERATOR_OBJ) $(EQUILSTATIC) 
PROJ_STATIC_DEPS=$(DENSESTATIC) $(CGSTATIC) $(PROJSTATIC)
POGS_STATIC_DEPS=$(POGSSTATIC) $(EQUILSTATIC) $(PROJSTATIC)
POGS_STATIC_DEPS += $(DENSESTATIC) $(PROXSTATIC)
POGS_ABSTRACT_STATIC_DEPS=$(POGSABSTRACTSTATIC) $(EQUILSTATIC) $(PROJSTATIC)
POGS_ABSTRACT_STATIC_DEPS += $(DENSESTATIC) $(SPARSESTATIC) $(PROXSTATIC) 
POGS_ABSTRACT_STATIC_DEPS += $(CGSTATIC) $(OPERATOR_OBJ)

.PHONY: default, all, libs, libok, libok_dense, libok_sparse, libprox
.PHONY: libpogs, libpogs_dense, libpogs_sparse, pogs_dense, pogs_sparse
.PHONY: libequil, equil, libprojector, projector, operator
default: cpu_dense
all: libs libequil libprojector libpogs

libs: libok libprox

libok: libok_dense libok_sparse

libpogs: $(POGSLIBS)

libpogs_dense: pogs equil projector $(LINSYSLIBS) libprox
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(POGS_STATIC_DEPS) $(LDFLAGS) 

libpogs_sparse: 
	#pogs equil projector $(LINSYSLIBS) libprox
	# mkdir -p $(OUT) \	
	# $(CXX) $(CXXFLAGS) -shared -o \
	# $(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED)  \
	# $(POGS_STATIC_DEPS) $(LDFLAGS)
	# $(OUT)$libok_dense_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	# $(OUT)$libok_sparse_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	# $(OUT)$libprox_$(DEVICETAG)$(PRECISION).$(SHARED) 

libpogs_abstract: pogs_abstract equil projector $(LINSYSLIBS) libprox
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(POGS_ABSTRACT_STATIC_DEPS) $(LDFLAGS) 

liboperator: operator $(DENSETARG) $(SPARSETARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(OPERATOR_OBJ) $(OPERATOR_STATIC_DEPS) $(LDFLAGS)

libprojector: projector cg $(DENSETARG) $(SPARSETARG) 
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(PROJ_STATIC_DEPS) $(LDFLAGS)

libcg: cg $(DENSETARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o\
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(CG_STATIC_DEPS) $(LDFLAGS) 

libequil: equil $(DENSETARG) $(SPARSETARG) operator
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(EQUIL_STATIC_DEPS) $(LDFLAGS)

libok_dense: $(DENSETARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(DENSESTATIC) $(LDFLAGS)

libok_sparse: $(SPARSETARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(SPARSESTATIC) $(SPARSE_STATIC_DEPS) $(LDFLAGS) 

libprox: $(PROXTARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o \
	$(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(PROXSTATIC) $(LDFLAGS)

cpu_dense: $(SRC)optkit_dense.c $(INCLUDE)optkit_dense.h
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(DENSESTATIC)	

gpu_dense: $(SRC)optkit_dense.cu $(INCLUDE)optkit_dense.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(DENSESTATIC)

cpu_sparse: $(SRC)optkit_sparse.c $(INCLUDE)optkit_sparse.h
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(SPARSESTATIC)	

gpu_sparse: $(SRC)optkit_sparse.cu $(INCLUDE)optkit_sparse.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(SPARSESTATIC)

cpu_prox: $(SRC)optkit_prox.cpp $(INCLUDE)optkit_prox.hpp
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PROXSTATIC)

gpu_prox: $(SRC)optkit_prox.cu $(INCLUDE)optkit_prox.hpp
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PROXSTATIC)

pogs_abstract: $(SRC)optkit_abstract_pogs.c $(INCLUDE)optkit_abstract_pogs.h
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(POGSABSTRACTSTATIC) \
	-I$(INCLUDE)operator/ -DABSTRACT_OPERATOR_BARE

pogs: $(SRC)optkit_pogs.c $(INCLUDE)optkit_pogs.h
	mkdir -p $(OUT) 
	$(CC) $(CCFLAGS) -I$(INCLUDE)operator/ $< -c -o $(POGSSTATIC)
	
equil: $(SRC)optkit_equilibration.c $(INCLUDE)optkit_equilibration.h
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(EQUILSTATIC) \
	-I$(INCLUDE)operator/ -DABSTRACT_OPERATOR_BARE

projector: $(SRC)optkit_projector.c $(INCLUDE)optkit_projector.h
	mkdir -p $(OUT)
	$(CC) $(CXXFLAGS) $< -c -o $(PROJSTATIC) -DABSTRACT_OPERATOR_BARE

operator: $(OPERATOR_SRC) 
	mkdir -p $(OUT)
	mkdir -p $(OUT)/operator
	$(CC) $(CCFLAGS) -I$(INCLUDE)operator/ $(OPSRC)dense.c  -c -o \
	$(OUT)$(OPERATOR)dense.o -DCOMPILE_ABSTRACT_OPERATOR
	$(CC) $(CCFLAGS) -I$(INCLUDE)operator/ $(OPSRC)sparse.c  -c -o \
	$(OUT)$(OPERATOR)sparse.o
	$(CC) $(CCFLAGS) -I$(INCLUDE)operator/ $(OPSRC)diagonal.c -c -o \
	$(OUT)$(OPERATOR)diagonal.o

cg: $(SRC)optkit_cg.c $(INCLUDE)optkit_projector.h
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $(CGSTATIC) -DABSTRACT_OPERATOR_BARE

.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)