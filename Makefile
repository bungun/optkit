# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
INCLUDE=$(OPTKITROOT)include/
PREFIX_OUT=$(OUT)optkit_

DEF_HDR=$(INCLUDE)optkit_defs.h
GPU_DEF_HDR=$(DEF_HDR) $(INCLUDE)optkit_defs_gpu.h

LINSYS=linsys/optkit_
LAINC=$(INCLUDE)$(LINSYS)
LASRC=$(SRC)$(LINSYS)
LAOUT=$(OUT)$(LINSYS)

OPERATOR=operator/optkit_operator_
OPINC=$(INCLUDE)$(OPERATOR)
OPSRC=$(SRC)$(OPERATOR)
OPOUT=$(OUT)$(OPERATOR)

CLUSTER=clustering/optkit_
CLUINC=$(INCLUDE)$(CLUSTER)
CLUSRC=$(SRC)$(CLUSTER)
CLUOUT=$(OUT)$(CLUSTER)

POGS=pogs/optkit_
POGSINC=$(INCLUDE)$(POGS)
POGSSRC=$(SRC)$(POGS)
POGSOUT=$(OUT)$(POGS)

IFLAGS=-I. -I$(INCLUDE) -I$(INCLUDE)external -I$(INCLUDE)linsys 
IFLAGS+=-I$(INCLUDE)operator -I$(INCLUDE)clustering

# C Flags
CC=gcc
CCFLAGS=-g -O3 -fPIC $(IFLAGS) -Wall -Wconversion -Wpedantic 
CCFLAGS+=-Wno-unused-function -std=c99
LDFLAGS_=-lstdc++ -lm

# C++ Flags
CXX=g++
CXXFLAGS=-g -O3 -fPIC $(IFLAGS) -Wall -Wconversion -Wpedantic 
CXXFLAGS+=-Wno-unused-function -std=c++11

# CUDA Flags
CUXX=nvcc
CUXXFLAGS=-arch=sm_50 -Xcompiler -fPIC $(IFLAGS) -std=c++11
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

BASE_TARG=$(DEVICETAG)_defs
VECTOR_TARG=$(DEVICETAG)_vector
DENSE_TARG=$(DEVICETAG)_dense
SPARSE_TARG=$(DEVICETAG)_sparse
LINSYS_TARGS=$(DENSE_TARG) $(SPARSE_TARG)

PROX_TARG=$(DEVICETAG)_prox
CLUSTER_TARG=$(DEVICETAG)_cluster


DENSE_HDR=$(LAINC)vector.h $(LAINC)matrix.h $(LAINC)blas.h $(LAINC)dense.h
SPARSE_HDR=$(LAINC)vector.h $(LAINC)sparse.h
ifneq ($(GPU), 0)
DENSE_HDR+=$(INCLUDE)optkit_thrust.hpp $(GPU_DEF_HDR)
SPARSE_HDR+=$(INCLUDE)optkit_thrust.hpp $(GPU_DEF_HDR)
else
DENSE_HDR+=$(DEF_HDR)
SPARSE_HDR+=$(DEF_HDR)
endif
LINSYS_HDR=$(DENSE_HDR) $(SPARSE_HDR)

OPERATOR_HDR=$(LINSYS_HDR) $(OPINC)dense.h $(OPINC)sparse.h $(OPINC)diagonal.h
OPERATOR_HDR+=$(INCLUDE)optkit_abstract_operator.h $(OPINC)transforms.h
CG_HDR=$(OPERATOR_HDR) $(INCLUDE)optkit_cg.h


BASE_OBJ=$(PREFIX_OUT)defs_$(LIBCONFIG).o
VECTOR_OBJ=$(LAOUT)vector_$(LIBCONFIG).o
DENSE_OBJ=$(LAOUT)vector_$(LIBCONFIG).o $(LAOUT)matrix_$(LIBCONFIG).o
DENSE_OBJ+=$(LAOUT)blas_$(LIBCONFIG).o $(LAOUT)dense_$(LIBCONFIG).o
SPARSE_OBJ=$(LAOUT)sparse_$(LIBCONFIG).o
PROX_OBJ=$(PREFIX_OUT)prox_$(LIBCONFIG).o

OPERATOR_OBJ=$(OPOUT)dense_$(LIBCONFIG).o $(OPOUT)sparse_$(LIBCONFIG).o 
OPERATOR_OBJ+=$(OPOUT)diagonal_$(LIBCONFIG).o 

CLUSTER_OBJ=$(CLUOUT)clustering_$(LIBCONFIG).o 
CLUSTER_OBJ+=$(CLUOUT)upsampling_vector_$(LIBCONFIG).o
CLUSTER_OBJ+=$(CLUOUT)upsampling_vector_common_$(LIBCONFIG).o
CLUSTER_OBJ+=$(CLUOUT)clustering_common_$(LIBCONFIG).o

CG_OBJ=$(PREFIX_OUT)cg_$(LIBCONFIG).o
PROJ_OBJ=$(PREFIX_OUT)projector_$(LIBCONFIG).o
PROJ_DIRECT_OBJ=$(PREFIX_OUT)projector_direct_$(LIBCONFIG).o
EQUIL_OBJ=$(PREFIX_OUT)equil_$(LIBCONFIG).o
EQUIL_DENSE_OBJ=$(PREFIX_OUT)equil_dense_$(LIBCONFIG).o

POGS_COMMON_OBJ=$(POGSOUT)pogs_common_$(LIBCONFIG).o 
POGS_OBJ=$(POGS_COMMON_OBJ) $(POGSOUT)pogs_$(LIBCONFIG).o
POGS_ABSTR_OBJ=$(POGS_COMMON_OBJ) $(POGSOUT)pogs_abstract_$(LIBCONFIG).o

SPARSE_STATIC_DEPS=$(BASE_OBJ) $(VECTOR_OBJ)
OPERATOR_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ) $(SPARSE_OBJ) $(OPERATOR_OBJ)
CG_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(CG_OBJ)
EQUIL_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(EQUIL_OBJ) 
PROJ_STATIC_DEPS=$(CG_STATIC_DEPS) $(PROJ_OBJ)
POGS_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ) $(PROX_OBJ) $(EQUIL_DENSE_OBJ) 
POGS_STATIC_DEPS+=$(PROJ_DIRECT_OBJ) $(POGS_OBJ) 
POGS_ABSTRACT_STATIC_DEPS=$(EQUIL_STATIC_DEPS) $(CG_OBJ) $(PROJ_OBJ) $(PROX_OBJ)
POGS_ABSTRACT_STATIC_DEPS+=$(POGS_ABSTR_OBJ)

POGS_DENSE_LIB_DEPS=equil_dense projector_direct $(DENSE_TARG) $(PROX_TARG)
POGS_SPARSE_LIB_DEPS=operator cg equil projector $(LINSYS_TARGS) $(PROX_TARG)
POGS_ABSTRACT_LIB_DEPS=operator cg equil projector $(LINSYS_TARGS) $(PROX_TARG)

.PHONY: default all libs libok libpogs pylibs
.PHONY: libpogs_abstract libpogs_sparse libpogs_dense libprojector libequil
.PHONY: libcg liboperator libcluster libprox libok_sparse libok_dense

.PHONY: pogs_abstract pogs pogs_abstract_ pogs_ pogs_common 
.PHONY: equil equil_dense projector projector_direct cg
.PHONY: operator dense_operator sparse_operator diagonal_operator
.PHONY: cpu_cluster gpu_cluster cpu_cluster_ gpu_cluster_ clustering_common
.PHONY: cpu_upsampling_vector_ gpu_upsampling_vector_ cpu_upsampling_vector 
.PHONY: gpu_upsampling_vector upsampling_vector_common 
.PHONY: cpu_prox gpu_prox cpu_sparse gpu_sparse
.PHONY: cpu_dense gpu_dense cpu_dense_ gpu_dense_ cpu_blas gpu_blas
.PHONY: cpu_matrix gpu_matrix cpu_vector gpu_vector cpu_defs gpu_defs

default: cpu_dense
all: libs liboperator libcg libequil libprojector libpogs libcluster
libs: libok libprox
libok: libok_dense libok_sparse
libpogs: libpogs_dense libpogs_abstract
pylibs: libpogs_dense libpogs_abstract libcluster

libpogs_abstract: $(OUT)libpogs_abstract_$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_abstract_$(LIBCONFIG).$(SHARED): pogs_abstract \
	$(POGS_ABSTRACT_LIB_DEPS) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs -shared -o $@ \
	$(POGS_ABSTRACT_STATIC_DEPS) $(LDFLAGS) 

libpogs_sparse: 
# libpogs_sparse: $(OUT)libpogs_sparse_$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_sparse_$(LIBCONFIG).$(SHARED): pogs $(POGS_SPARSE_LIB_DEPS) \
	$(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(POGS_STATIC_DEPS) $(SPARSE_OBJ) $(LDFLAGS)

libpogs_dense: $(OUT)libpogs_dense_$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_dense_$(LIBCONFIG).$(SHARED): pogs $(POGS_DENSE_LIB_DEPS) \
	$(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(POGS_STATIC_DEPS) $(LDFLAGS) 

libprojector: $(OUT)libprojector_$(LIBCONFIG).$(SHARED)
$(OUT)libprojector_$(LIBCONFIG).$(SHARED): projector operator cg $(DENSE_TARG) \
	$(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(PROJ_STATIC_DEPS) $(LDFLAGS)

libequil: $(OUT)libequil_$(LIBCONFIG).$(SHARED)
$(OUT)libequil_$(LIBCONFIG).$(SHARED): equil $(DENSE_TARG) $(SPARSE_TARG) \
	operator $(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(EQUIL_STATIC_DEPS) $(LDFLAGS)

libcg: $(OUT)libcg_$(LIBCONFIG).$(SHARED)
$(OUT)libcg_$(LIBCONFIG).$(SHARED): cg $(DENSE_TARG) $(SPARSE_TARG) operator \
	$(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(CG_STATIC_DEPS) $(LDFLAGS) 

liboperator: $(OUT)liboperator_$(LIBCONFIG).$(SHARED)
$(OUT)liboperator_$(LIBCONFIG).$(SHARED): operator $(DENSE_TARG) \
	$(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(OPERATOR_STATIC_DEPS) $(LDFLAGS)

libcluster: $(OUT)libcluster_$(LIBCONFIG).$(SHARED)
$(OUT)libcluster_$(LIBCONFIG).$(SHARED): $(CLUSTER_TARG) $(DENSE_TARG) \
	$(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(CLUSTER_OBJ) $(DENSE_OBJ) $(BASE_OBJ) $(LDFLAGS)	

libprox: $(OUT)libprox_$(LIBCONFIG).$(SHARED)
$(OUT)libprox_$(LIBCONFIG).$(SHARED): $(PROX_TARG) $(VECTOR_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(PROX_OBJ) $(VECTOR_OBJ) $(BASE_OBJ) $(LDFLAGS)

libok_sparse: $(OUT)libok_sparse_$(LIBCONFIG).$(SHARED)
$(OUT)libok_sparse_$(LIBCONFIG).$(SHARED): $(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(SPARSE_OBJ) $(SPARSE_STATIC_DEPS) $(LDFLAGS) 

libok_dense: $(OUT)libok_dense_$(LIBCONFIG).$(SHARED)
$(OUT)libok_dense_$(LIBCONFIG).$(SHARED): $(DENSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(DENSE_OBJ) $(BASE_OBJ) $(LDFLAGS)

pogs_abstract: pogs_common pogs_abstract_
pogs: pogs_common pogs_

pogs_abstract_: $(POGSOUT)pogs_abstract_$(LIBCONFIG).o
$(POGSOUT)pogs_abstract_$(LIBCONFIG).o: $(POGSSRC)pogs_abstract.c \
	$(POGSINC)pogs_abstract.h $(INCLUDE)optkit_equilibration.h \
	$(INCLUDE)optkit_projector.h
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@

pogs_: $(POGSOUT)pogs_$(LIBCONFIG).o
$(POGSOUT)pogs_$(LIBCONFIG).o: $(POGSSRC)pogs.c $(POGSINC)pogs.h \
	$(INCLUDE)optkit_equilibration.h $(INCLUDE)optkit_projector.h
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@	

pogs_common: $(POGSOUT)pogs_common_$(LIBCONFIG).o
$(POGSOUT)pogs_common_$(LIBCONFIG).o: $(POGSSRC)pogs_common.c \
	$(POGSINC)pogs_common.h $(DENSE_HDR) $(INCLUDE)optkit_prox.hpp
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@

equil: $(EQUIL_OBJ)
$(EQUIL_OBJ): $(SRC)optkit_equilibration.c $(INCLUDE)optkit_equilibration.h \
	$(OPERATOR_HDR)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@

equil_dense: $(EQUIL_DENSE_OBJ)
$(EQUIL_DENSE_OBJ): $(SRC)optkit_equilibration.c \
	$(INCLUDE)optkit_equilibration.h $(DENSE_HDR)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@ -DOPTKIT_NO_OPERATOR_EQUIL

projector: $(PROJ_OBJ)
$(PROJ_OBJ): $(SRC)optkit_projector.c $(INCLUDE)optkit_projector.h $(CG_HDR)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@

projector_direct: $(PROJ_DIRECT_OBJ)
$(PROJ_DIRECT_OBJ): $(SRC)optkit_projector.c $(INCLUDE)optkit_projector.h \
	$(DENSE_HDR)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@ -DOPTKIT_NO_INDIRECT_PROJECTOR

cg: $(CG_OBJ)
$(CG_OBJ): $(SRC)optkit_cg.c $(INCLUDE)optkit_cg.h $(OPERATOR_HDR) 
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@

operator: dense_operator sparse_operator diagonal_operator
dense_operator: $(OUT)$(OPERATOR)dense_$(LIBCONFIG).o
$(OUT)$(OPERATOR)dense_$(LIBCONFIG).o: $(OPSRC)dense.c $(OPINC)dense.h \
	$(INCLUDE)optkit_abstract_operator.h $(OPINC)transforms.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)operator
	$(CC) $(CCFLAGS) $< -c -o $@

sparse_operator: $(OUT)$(OPERATOR)sparse_$(LIBCONFIG).o
$(OUT)$(OPERATOR)sparse_$(LIBCONFIG).o: $(OPSRC)sparse.c $(OPINC)sparse.h \
	$(INCLUDE)optkit_abstract_operator.h $(OPINC)transforms.h $(SPARSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)operator
	$(CC) $(CCFLAGS) $< -c -o $@

diagonal_operator: $(OUT)$(OPERATOR)diagonal_$(LIBCONFIG).o
$(OUT)$(OPERATOR)diagonal_$(LIBCONFIG).o: $(OPSRC)diagonal.c \
	$(OPINC)diagonal.h $(INCLUDE)optkit_abstract_operator.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)operator
	$(CC) $(CCFLAGS) $< -c -o $@

cpu_cluster: cpu_upsampling_vector clustering_common cpu_cluster_ 
gpu_cluster: gpu_upsampling_vector clustering_common gpu_cluster_ 

cpu_cluster_: $(CLUOUT)clustering_cpu$(PRECISION).o
$(CLUOUT)clustering_cpu$(PRECISION).o: $(CLUSRC)clustering.c 
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_cluster_: $(CLUOUT)clustering_gpu$(PRECISION).o
$(CLUOUT)clustering_gpu$(PRECISION).o: $(CLUSRC)clustering.cu
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

clustering_common: $(CLUOUT)clustering_common_$(LIBCONFIG).o
$(CLUOUT)clustering_common_$(LIBCONFIG).o: $(CLUSRC)clustering_common.c \
	$(CLUINC)clustering.h $(DEVICETAG)_upsampling_vector
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

cpu_upsampling_vector: upsampling_vector_common cpu_upsampling_vector_ 
gpu_upsampling_vector: upsampling_vector_common gpu_upsampling_vector_

cpu_upsampling_vector: $(CLUOUT)upsampling_vector_cpu$(PRECISION).o
$(CLUOUT)upsampling_vector_cpu$(PRECISION).o: $(CLUSRC)upsampling_vector.c
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_upsampling_vector: $(CLUOUT)upsampling_vector_gpu$(PRECISION).o
$(CLUOUT)upsampling_vector_gpu$(PRECISION).o: $(CLUSRC)upsampling_vector.cu
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

upsampling_vector_common: $(CLUOUT)upsampling_vector_common_$(LIBCONFIG).o
$(CLUOUT)upsampling_vector_common_$(LIBCONFIG).o: \
	$(CLUSRC)upsampling_vector_common.c $(CLUINC)upsampling_vector.h \
	$(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

cpu_prox: $(PREFIX_OUT)prox_cpu$(PRECISION).o
$(PREFIX_OUT)prox_cpu$(PRECISION).o: $(SRC)optkit_prox.cpp \
	$(INCLUDE)optkit_prox.hpp $(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $@

gpu_prox: $(PREFIX_OUT)prox_gpu$(PRECISION).o
$(PREFIX_OUT)prox_gpu$(PRECISION).o:$(SRC)optkit_prox.cu \
	$(INCLUDE)optkit_prox.hpp $(INCLUDE)optkit_thrust.hpp $(LAINC)vector.h \
	$(GPU_DEF_HDR)
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_sparse: $(LAOUT)sparse_cpu$(PRECISION).o
$(LAOUT)sparse_cpu$(PRECISION).o: $(LASRC)sparse.cpp $(LAINC)sparse.h \
	$(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CXX) $(CXXFLAGS) $< -c -o $@

gpu_sparse: $(LAOUT)sparse_gpu$(PRECISION).o
$(LAOUT)sparse_gpu$(PRECISION).o: $(LASRC)sparse.cu $(LAINC)sparse.h \
	$(LAINC)vector.h $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_dense: cpu_vector cpu_matrix cpu_blas cpu_dense_
gpu_dense: gpu_vector gpu_matrix gpu_blas gpu_dense_

cpu_dense_: $(LAOUT)dense_cpu$(PRECISION).o
$(LAOUT)dense_cpu$(PRECISION).o: $(LASRC)dense.c $(LAINC)dense.h \
	$(LAINC)blas.h $(LAINC)matrix.h $(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_dense_: $(LAOUT)dense_gpu$(PRECISION).o
$(LAOUT)dense_gpu$(PRECISION).o: $(LASRC)dense.cu $(LAINC)dense.h \
	$(LAINC)blas.h $(LAINC)matrix.h $(LAINC)vector.h $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_blas: $(LAOUT)blas_cpu$(PRECISION).o
$(LAOUT)blas_cpu$(PRECISION).o: $(LASRC)blas.c $(LAINC)blas.h \
	$(LAINC)matrix.h $(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_blas: $(LAOUT)blas_gpu$(PRECISION).o
$(LAOUT)blas_gpu$(PRECISION).o: $(LASRC)blas.cu $(LAINC)blas.h \
	$(LAINC)matrix.h $(LAINC)vector.h $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_matrix: $(LAOUT)matrix_cpu$(PRECISION).o
$(LAOUT)matrix_cpu$(PRECISION).o: $(LASRC)matrix.cpp $(LAINC)matrix.h \
	$(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CXX) $(CXXFLAGS) $< -c -o $@

gpu_matrix: $(LAOUT)matrix_gpu$(PRECISION).o
$(LAOUT)matrix_gpu$(PRECISION).o: $(LASRC)matrix.cu $(LAINC)matrix.h \
	$(LAINC)vector.h $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_vector: $(LAOUT)vector_cpu$(PRECISION).o
$(LAOUT)vector_cpu$(PRECISION).o: $(LASRC)vector.cpp $(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CXX) $(CXXFLAGS) $< -c -o $@

gpu_vector: $(LAOUT)vector_gpu$(PRECISION).o
$(LAOUT)vector_gpu$(PRECISION).o: $(LASRC)vector.cu $(LAINC)vector.h \
	$(INCLUDE)optkit_thrust.hpp $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_defs: $(PREFIX_OUT)defs_cpu$(PRECISION).o
$(PREFIX_OUT)defs_cpu$(PRECISION).o: $(SRC)optkit_defs.c $(DEF_HDR)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_defs: $(PREFIX_OUT)defs_gpu$(PRECISION).o
$(PREFIX_OUT)defs_gpu$(PRECISION).o: $(SRC)optkit_defs.cu $(GPU_DEF_HDR)
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)