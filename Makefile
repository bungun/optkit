# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
INCLUDE=$(OPTKITROOT)include/
PREFIX_OUT=$(OUT)optkit_

# C++ Flags
CXX=gcc
CXXFLAGS= -g -O3 -fPIC -I. -I./include -I./include/external -Wall -Wconversion
LDFLAGS_=-lstdc++ -lm

# CUDA Flags
CUXX=nvcc
CUXXFLAGS=-arch=sm_50 -Xcompiler -fPIC -I. -I./include -I./include/external
CULDFLAGS_=-lstdc++ -lm

# Check system args
ifeq ($(shell uname -s), Darwin)
LDFLAGS_ += -framework Accelerate
CULDFLAGS_ += -L/usr/local/cuda/lib 
SHARED=dylib
ifdef USE_OPENMP
CXXFLAGS += -fopenmp=libopenmp
endif
else
LDFLAGS_ += -lblas
CULDFLAGS_ += -L/usr/local/cuda/lib64 
SHARED=so
ifdef USE_OPENMP
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

ORDER=
ifneq ($(ROWMAJOR), 0)
ORDER=row_
OPT_FLAGS += -DOPTKIT_ROWMAJOR
else
ifneq ($(COLMAJOR), 0)
ORDER=col_
OPT_FLAGS += -DOPTKIT_COLMAJOR
endif
endif

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
PROJSTATIC=$(PREFIX_OUT)projector_$(DEVICETAG)$(PRECISION).o$
EQUILSTATIC=$(PREFIX_OUT)equil_$(DEVICETAG)$(PRECISION).o$
POGSSTATIC=$(PREFIX_OUT)pogs_$(DEVICETAG)$(PRECISION).o

POGS_STATIC_DEPS=$(POGSSTATIC) $(EQUILSTATIC) $(PROJSTATIC)
POGS_STATIC_DEPS += $(DENSESTATIC) $(PROXSTATIC)
PROJ_STATIC_DEPS=$(DENSESTATIC) $(PROJSTATIC)
EQUIL_STATIC_DEPS=$(DENSESTATIC) $(EQUILSTATIC)
SPARSE_STATIC_DEPS=$(DENSESTATIC)
ifneq ($(SPARSE), 0)
PROJ_STATIC_DEPS += $(SPARSESTATIC)
EQUIL_STATIC_DEPS += $(SPARSESTATIC)
POGS_STATIC_DEPS += $(SPARSESTATIC) 
endif

.PHONY: default, all, libs, libok, libok_dense, libok_sparse, libprox
.PHONY: libpogs, libpogs_dense, libpogs_sparse, pogs_dense, pogs_sparse
.PHONY: libequil, equil, libprojector, projector 
default: cpu_dense
all: libs libequil libprojector libpogs

libs: libok libprox

libok: libok_dense libok_sparse

libpogs: $(POGSLIBS)

libpogs_dense: pogs equil projector $(LINSYSLIBS) libprox
	mkdir -p $(OUT)	
	$(CXX) $(CXXFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(POGS_STATIC_DEPS) $(LDFLAGS) 


libpogs_sparse: 
	#pogs equil projector $(LINSYSLIBS) libprox
	# mkdir -p $(OUT) \	
	# $(CXX) $(CXXFLAGS) -shared -o \
	# $(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	# $(POGS_STATIC_DEPS) $(LDFLAGS)
	# $(OUT)$libok_dense_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	# $(OUT)$libok_sparse_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	# $(OUT)$libprox_$(DEVICETAG)$(PRECISION).$(SHARED) 

libprojector: projector $(DENSETARG) $(SPARSETARG)
	mkdir -p $(OUT)	
	$(CXX) $(CXXFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(PROJ_STATIC_DEPS) $(LDFLAGS)


libequil: equil $(DENSETARG) $(SPARSETARG)
	mkdir -p $(OUT)	
	$(CXX) $(CXXFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED)  \
	$(EQUIL_STATIC_DEPS) $(LDFLAGS)


libok_dense: $(DENSETARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(DENSESTATIC) $(LDFLAGS)

libok_sparse: $(SPARSETARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o \
	$(OUT)$@_$(ORDER)$(DEVICETAG)$(PRECISION).$(SHARED) \
	$(SPARSESTATIC) $(SPARSE_STATIC_DEPS ) $(LDFLAGS) 


libprox: $(PROXTARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o $(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) $(PROXSTATIC) $(LDFLAGS)

cpu_dense: $(SRC)optkit_dense.c $(INCLUDE)optkit_dense.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(DENSESTATIC)	

gpu_dense: $(SRC)optkit_dense.cu $(INCLUDE)optkit_dense.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(DENSESTATIC)

cpu_sparse: $(SRC)optkit_sparse.c $(INCLUDE)optkit_sparse.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(SPARSESTATIC)	

gpu_sparse: $(SRC)optkit_sparse.cu $(INCLUDE)optkit_sparse.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(SPARSESTATIC)

cpu_prox: $(SRC)optkit_prox.cpp $(INCLUDE)optkit_prox.hpp
	mkdir -p $(OUT)
	g++ $(CXXFLAGS) $< -c -o $(PROXSTATIC)

gpu_prox: $(SRC)optkit_prox.cu $(INCLUDE)optkit_prox.hpp
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PROXSTATIC)

pogs: $(SRC)optkit_pogs.c $(INCLUDE)optkit_pogs.h
	mkdir -p $(OUT) 
	$(CXX) $(CXXFLAGS) $< -c -o $(POGSSTATIC)
	
equil: $(SRC)optkit_equilibration.c $(INCLUDE)optkit_equilibration.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(EQUILSTATIC)

projector: $(SRC)optkit_projector.c $(INCLUDE)optkit_projector.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PROJSTATIC)



.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)
