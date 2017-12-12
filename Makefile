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

ANDERSON=anderson/optkit_
ANDINC=$(INCLUDE)$(ANDERSON)
ANDSRC=$(SRC)$(ANDERSON)
ANDOUT=$(OUT)$(ANDERSON)

POGS=pogs/optkit_pogs_
POGSINC=$(INCLUDE)$(POGS)
POGSSRC=$(SRC)$(POGS)
POGSOUT=$(OUT)$(POGS)

IFLAGS=-I. -I$(INCLUDE) -I$(INCLUDE)external -I$(INCLUDE)linsys 
IFLAGS+=-I$(INCLUDE)anderson -I$(INCLUDE)operator -I$(INCLUDE)clustering 
IFLAGS+=-I$(INCLUDE)pogs


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

ifndef NO_LAPACK
LDFLAGS_+=-llapack 
CULDFLAGS_+=-lcusolver
endif

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

LIBCONFIG=_$(DEVICETAG)$(PRECISION)

BASE_TARG=$(DEVICETAG)_defs
VECTOR_TARG=$(DEVICETAG)_vector
DENSE_TARG=$(DEVICETAG)_dense
SPARSE_TARG=$(DEVICETAG)_sparse
LINSYS_TARGS=$(DENSE_TARG) $(SPARSE_TARG)

PROX_TARG=$(DEVICETAG)_prox
CLUSTER_TARG=$(DEVICETAG)_cluster


DENSE_HDR=$(LAINC)vector.h $(LAINC)matrix.h $(LAINC)blas.h $(LAINC)lapack.h  
DENSE_HDR+=$(LAINC)dense.h
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

# POGS_BASE_HDR=$(DENSE_HDR) $(INCLUDE)optkit_prox.hpp $(POGSINC)datatypes.h 
# POGS_BASE_HDR+=$(POGSINC)impl_common.h $(POGSINC)adaptive_rho.h 
# POGS_BASE_HDR+=$(INCLUDE)optkit_projector.h $(INCLUDE)optkit_equilibration.h
# POGS_ABSTRACT_HDR=$(POGS_BASE_HDR) $(POGSINC)impl_abstract.h
# POGS_ABSTRACT_HDR+=
# POGS_SPARSE_HDR=$(POGS_BASE_HDR) $(POGSINC)impl_sparse.h
# POGS_DENSE_HDR=$(POGS_BASE_HDR) $(POGSINC)impl_dense.h

BASE_OBJ=$(PREFIX_OUT)defs$(LIBCONFIG).o
VECTOR_OBJ=$(LAOUT)vector$(LIBCONFIG).o
DENSE_OBJ=$(LAOUT)vector$(LIBCONFIG).o $(LAOUT)matrix$(LIBCONFIG).o
DENSE_OBJ+=$(LAOUT)blas$(LIBCONFIG).o  $(LAOUT)lapack$(LIBCONFIG).o
DENSE_OBJ+=$(LAOUT)dense$(LIBCONFIG).o
SPARSE_OBJ=$(LAOUT)sparse$(LIBCONFIG).o
PROX_OBJ=$(PREFIX_OUT)prox$(LIBCONFIG).o

ANDERSON_OBJ=$(ANDOUT)anderson$(LIBCONFIG).o 
ANDERSON_OBJ+=$(ANDOUT)anderson_difference$(LIBCONFIG).o
ANDERSON_OBJ+=$(ANDOUT)anderson_fused$(LIBCONFIG).o
ANDERSON_OBJ+=$(ANDOUT)anderson_fused_diff$(LIBCONFIG).o

OPERATOR_OBJ=$(OPOUT)dense$(LIBCONFIG).o $(OPOUT)sparse$(LIBCONFIG).o 
OPERATOR_OBJ+=$(OPOUT)diagonal$(LIBCONFIG).o 

CLUSTER_OBJ=$(CLUOUT)clustering$(LIBCONFIG).o 
CLUSTER_OBJ+=$(CLUOUT)upsampling_vector$(LIBCONFIG).o
CLUSTER_OBJ+=$(CLUOUT)upsampling_vector_common$(LIBCONFIG).o
CLUSTER_OBJ+=$(CLUOUT)clustering_common$(LIBCONFIG).o

CG_OBJ=$(PREFIX_OUT)cg$(LIBCONFIG).o
PROJ_OBJ=$(PREFIX_OUT)projector$(LIBCONFIG).o
PROJ_DIRECT_OBJ=$(PREFIX_OUT)projector_direct$(LIBCONFIG).o
EQUIL_OBJ=$(PREFIX_OUT)equil$(LIBCONFIG).o
EQUIL_DENSE_OBJ=$(PREFIX_OUT)equil_dense$(LIBCONFIG).o

# POGS_COMMON_OBJ=$(POGSOUT)pogs_common$(LIBCONFIG).o 
# POGS_OBJ=$(POGS_COMMON_OBJ) $(POGSOUT)pogs$(LIBCONFIG).o
# POGS_ABSTR_OBJ=$(POGS_COMMON_OBJ) $(POGSOUT)pogs_abstract$(LIBCONFIG).o

POGS_DENSE_OBJ=$(POGSOUT)dense$(LIBCONFIG).o
POGS_SPARSE_OBJ=$(POGSOUT)sparse$(LIBCONFIG).o
POGS_ABSTRACT_OBJ=$(POGSOUT)abstract$(LIBCONFIG).o

SPARSE_STATIC_DEPS=$(BASE_OBJ) $(VECTOR_OBJ)
ANDERSON_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ)
OPERATOR_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ) $(SPARSE_OBJ) $(OPERATOR_OBJ)
CG_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(CG_OBJ)
EQUIL_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(EQUIL_OBJ) 
PROJ_STATIC_DEPS=$(CG_STATIC_DEPS) $(PROJ_OBJ)

# POGS_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ) $(PROX_OBJ) $(EQUIL_DENSE_OBJ) 
# POGS_STATIC_DEPS+=$(PROJ_DIRECT_OBJ) $(POGS_OBJ) 
# POGS_ABSTRACT_STATIC_DEPS=$(EQUIL_STATIC_DEPS) $(CG_OBJ) $(PROJ_OBJ) $(PROX_OBJ)
# POGS_ABSTRACT_STATIC_DEPS+=$(POGS_ABSTR_OBJ)

POGS_DENSE_STATIC_DEPS=$(BASE_OBJ) $(DENSE_OBJ) $(PROX_OBJ) $(EQUIL_DENSE_OBJ) 
POGS_DENSE_STATIC_DEPS+=$(PROJ_DIRECT_OBJ) $(ANDERSON_OBJ) $(POGS_DENSE_OBJ) 
POGS_SPARSE_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(PROX_OBJ) $(CG_OBJ) $(PROJ_OBJ)
POGS_SPARSE_STATIC_DEPS+=$(EQUIL_OBJ) $(ANDERSON_OBJ) $(POGS_SPARSE_OBJ)
POGS_ABSTRACT_STATIC_DEPS=$(OPERATOR_STATIC_DEPS) $(PROX_OBJ) $(CG_OBJ) $(PROJ_OBJ)
POGS_ABSTRACT_STATIC_DEPS+=$(EQUIL_OBJ) $(ANDERSON_OBJ) $(POGS_ABSTRACT_OBJ)

POGS_DENSE_LIB_DEPS=$(BASE_TARG) $(DENSE_TARG) $(PROX_TARG) 
POGS_DENSE_LIB_DEPS+=equil_dense projector_direct anderson
POGS_SPARSE_LIB_DEPS=$(BASE_TARG) $(LINSYS_TARGS) $(PROX_TARG) 
POGS_SPARSE_LIB_DEPS+=operator cg equil projector anderson 
POGS_ABSTRACT_LIB_DEPS=$(BASE_TARG) $(LINSYS_TARGS) $(PROX_TARG) 
POGS_ABSTRACT_LIB_DEPS+=operator cg equil projector anderson 

.PHONY: default all libs libok libpogs pylibs
.PHONY: libpogs_abstract libpogs_sparse libpogs_dense libprojector libequil
.PHONY: libcg liboperator libanderson libcluster 
.PHONY: libprox libok_sparse libok_dense

.PHONY: pogs_abstract pogs_dense pogs_#pogs_abstract_ pogs_common 
.PHONY: equil equil_dense projector projector_direct cg anderson 
.PHONY: operator dense_operator sparse_operator diagonal_operator
.PHONY: cpu_cluster gpu_cluster cpu_cluster_ gpu_cluster_ clustering_common
.PHONY: cpu_upsampling_vector_ gpu_upsampling_vector_ cpu_upsampling_vector 
.PHONY: gpu_upsampling_vector upsampling_vector_common 
.PHONY: anderson anderson_fused anderson_fused_ anderson_explicit
.PHONY: anderson_fused_diff anderson_fused_diff_ anderson_difference
.PHONY: cpu_prox gpu_prox cpu_sparse gpu_sparse
.PHONY: cpu_dense gpu_dense cpu_dense_ gpu_dense_ 
.PHONY: cpu_lapack gpu_lapack cpu_blas gpu_blas 
.PHONY: cpu_matrix gpu_matrix cpu_vector gpu_vector cpu_defs gpu_defs

default: cpu_dense
all: libs liboperator libcg libequil libprojector libpogs libcluster libanderson
libs: libok libprox
libok: libok_dense libok_sparse
libpogs: libpogs_dense libpogs_abstract
pylibs: libpogs_dense libpogs_abstract libcluster

libpogs_abstract: $(OUT)libpogs_abstract$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_abstract$(LIBCONFIG).$(SHARED): pogs_abstract \
	$(POGS_ABSTRACT_LIB_DEPS)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(POGS_ABSTRACT_STATIC_DEPS) $(LDFLAGS) 

libpogs_sparse: $(OUT)libpogs_sparse$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_sparse$(LIBCONFIG).$(SHARED): pogs_sparse \
	$(POGS_SPARSE_LIB_DEPS) 
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(POGS_SPARSE_STATIC_DEPS)  $(LDFLAGS)

libpogs_dense: $(OUT)libpogs_dense$(LIBCONFIG).$(SHARED)
$(OUT)libpogs_dense$(LIBCONFIG).$(SHARED): pogs_dense $(POGS_DENSE_LIB_DEPS) 
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(POGS_DENSE_STATIC_DEPS) $(LDFLAGS) 

libprojector: $(OUT)libprojector$(LIBCONFIG).$(SHARED)
$(OUT)libprojector$(LIBCONFIG).$(SHARED): projector operator cg $(DENSE_TARG) \
	$(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(PROJ_STATIC_DEPS) $(LDFLAGS)

libequil: $(OUT)libequil$(LIBCONFIG).$(SHARED)
$(OUT)libequil$(LIBCONFIG).$(SHARED): equil $(DENSE_TARG) $(SPARSE_TARG) \
	operator $(BASE_TARG)
	mkdir -p $(OUT)	
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(EQUIL_STATIC_DEPS) $(LDFLAGS)

libcg: $(OUT)libcg$(LIBCONFIG).$(SHARED)
$(OUT)libcg$(LIBCONFIG).$(SHARED): cg $(DENSE_TARG) $(SPARSE_TARG) operator \
	$(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(CG_STATIC_DEPS) $(LDFLAGS) 

liboperator: $(OUT)liboperator$(LIBCONFIG).$(SHARED)
$(OUT)liboperator$(LIBCONFIG).$(SHARED): operator $(DENSE_TARG) \
	$(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(OPERATOR_STATIC_DEPS) $(LDFLAGS)

libcluster: $(OUT)libcluster$(LIBCONFIG).$(SHARED)
$(OUT)libcluster$(LIBCONFIG).$(SHARED): $(CLUSTER_TARG) $(DENSE_TARG) \
	$(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(CLUSTER_OBJ) $(DENSE_OBJ) $(BASE_OBJ) $(LDFLAGS)	

libanderson: $(OUT)libanderson$(LIBCONFIG).$(SHARED)
$(OUT)libanderson$(LIBCONFIG).$(SHARED): anderson $(DENSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(ANDERSON_OBJ) $(ANDERSON_STATIC_DEPS) $(LDFLAGS)

libprox: $(OUT)libprox$(LIBCONFIG).$(SHARED)
$(OUT)libprox$(LIBCONFIG).$(SHARED): $(PROX_TARG) $(VECTOR_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(PROX_OBJ) $(VECTOR_OBJ) $(BASE_OBJ) $(LDFLAGS)

libok_sparse: $(OUT)libok_sparse$(LIBCONFIG).$(SHARED)
$(OUT)libok_sparse$(LIBCONFIG).$(SHARED): $(SPARSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(SPARSE_OBJ) $(SPARSE_STATIC_DEPS) $(LDFLAGS) 

libok_dense: $(OUT)libok_dense$(LIBCONFIG).$(SHARED)
$(OUT)libok_dense$(LIBCONFIG).$(SHARED): $(DENSE_TARG) $(BASE_TARG)
	mkdir -p $(OUT)
	$(CC) $(CCFLAGS) -shared -o $@ \
	$(DENSE_OBJ) $(BASE_OBJ) $(LDFLAGS)

# pogs_abstract: pogs_common pogs_abstract_
# pogs: pogs_common pogs_

# pogs_abstract_: $(POGSOUT)pogs_abstract$(LIBCONFIG).o
# $(POGSOUT)pogs_abstract$(LIBCONFIG).o: $(POGSSRC)pogs_abstract.c \
# 	$(POGSINC)pogs_abstract.h $(INCLUDE)optkit_equilibration.h \
# 	$(INCLUDE)optkit_projector.h
# 	mkdir -p $(OUT) 
# 	mkdir -p $(OUT)pogs 	
# 	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@

# pogs_: $(POGSOUT)pogs$(LIBCONFIG).o
# $(POGSOUT)pogs$(LIBCONFIG).o: $(POGSSRC)pogs.c $(POGSINC)pogs.h \
# 	$(INCLUDE)optkit_equilibration.h $(INCLUDE)optkit_projector.h
# 	mkdir -p $(OUT) 
# 	mkdir -p $(OUT)pogs 	
# 	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@	

# pogs_common: $(POGSOUT)pogs_common$(LIBCONFIG).o
# $(POGSOUT)pogs_common$(LIBCONFIG).o: $(POGSSRC)pogs_common.c \
# 	$(POGSINC)pogs_common.h $(DENSE_HDR) $(INCLUDE)optkit_prox.hpp
# 	mkdir -p $(OUT) 
# 	mkdir -p $(OUT)pogs 	
# 	$(CC) $(CCFLAGS) -I$(INCLUDE)pogs $< -c -o $@

pogs_abstract: $(POGSOUT)abstract$(LIBCONFIG).o
$(POGSOUT)abstract$(LIBCONFIG).o: $(POGSSRC)generic.c 
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -DOK_COMPILE_POGS_ABSTRACT \
	-I$(INCLUDE)pogs -I$(INCLUDE)operator $< -c -o $@

pogs_sparse: $(POGSOUT)dense$(LIBCONFIG).o
$(POGSOUT)sparse$(LIBCONFIG).o: $(POGSSRC)generic.c 
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -DOK_COMPILE_POGS_SPARSE -I$(INCLUDE)pogs $< -c -o $@

pogs_dense: $(POGSOUT)dense$(LIBCONFIG).o
$(POGSOUT)dense$(LIBCONFIG).o: $(POGSSRC)generic.c 
	mkdir -p $(OUT) 
	mkdir -p $(OUT)pogs 	
	$(CC) $(CCFLAGS) -DOK_COMPILE_POGS_DENSE -I$(INCLUDE)pogs $< -c -o $@

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
dense_operator: $(OUT)$(OPERATOR)dense$(LIBCONFIG).o
$(OUT)$(OPERATOR)dense$(LIBCONFIG).o: $(OPSRC)dense.c $(OPINC)dense.h \
	$(INCLUDE)optkit_abstract_operator.h $(OPINC)transforms.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)operator
	$(CC) $(CCFLAGS) $< -c -o $@

sparse_operator: $(OUT)$(OPERATOR)sparse$(LIBCONFIG).o
$(OUT)$(OPERATOR)sparse$(LIBCONFIG).o: $(OPSRC)sparse.c $(OPINC)sparse.h \
	$(INCLUDE)optkit_abstract_operator.h $(OPINC)transforms.h $(SPARSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)operator
	$(CC) $(CCFLAGS) $< -c -o $@

diagonal_operator: $(OUT)$(OPERATOR)diagonal$(LIBCONFIG).o
$(OUT)$(OPERATOR)diagonal$(LIBCONFIG).o: $(OPSRC)diagonal.c \
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

clustering_common: $(CLUOUT)clustering_common$(LIBCONFIG).o
$(CLUOUT)clustering_common$(LIBCONFIG).o: $(CLUSRC)clustering_common.c \
	$(CLUINC)clustering.h $(DEVICETAG)_upsampling_vector
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

cpu_upsampling_vector: upsampling_vector_common cpu_upsampling_vector_ 
gpu_upsampling_vector: upsampling_vector_common gpu_upsampling_vector_

cpu_upsampling_vector_: $(CLUOUT)upsampling_vector_cpu$(PRECISION).o
$(CLUOUT)upsampling_vector_cpu$(PRECISION).o: $(CLUSRC)upsampling_vector.c
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_upsampling_vector_: $(CLUOUT)upsampling_vector_gpu$(PRECISION).o
$(CLUOUT)upsampling_vector_gpu$(PRECISION).o: $(CLUSRC)upsampling_vector.cu
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

upsampling_vector_common: $(CLUOUT)upsampling_vector_common$(LIBCONFIG).o
$(CLUOUT)upsampling_vector_common$(LIBCONFIG).o: \
	$(CLUSRC)upsampling_vector_common.c $(CLUINC)upsampling_vector.h \
	$(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)clustering/
	$(CC) $(CCFLAGS) $< -c -o $@

anderson: anderson_fused anderson_fused_diff
anderson_fused: anderson_explicit anderson_fused_
anderson_fused_diff: anderson_difference anderson_fused_diff_ 

anderson_fused_: $(ANDOUT)anderson_fused$(LIBCONFIG).o
$(ANDOUT)anderson_fused$(LIBCONFIG).o: $(ANDSRC)anderson_fused.c \
	$(ANDINC)anderson.h $(ANDINC)anderson_reductions.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)anderson/
	$(CC) $(CCFLAGS) $< -c -o $@

anderson_fused_diff_: $(ANDOUT)anderson_fused_diff$(LIBCONFIG).o
$(ANDOUT)anderson_fused_diff$(LIBCONFIG).o: $(ANDSRC)anderson_fused_diff.c \
	$(ANDINC)anderson_difference.h $(ANDINC)anderson_reductions.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)anderson/
	$(CC) $(CCFLAGS) $< -c -o $@

anderson_explicit: $(ANDOUT)anderson$(LIBCONFIG).o
$(ANDOUT)anderson$(LIBCONFIG).o: $(ANDSRC)anderson.c \
	$(ANDINC)anderson.h $(ANDINC)anderson_reductions.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)anderson/
	$(CC) $(CCFLAGS) $< -c -o $@

anderson_difference: $(ANDOUT)anderson_difference$(LIBCONFIG).o
$(ANDOUT)anderson_difference$(LIBCONFIG).o: $(ANDSRC)anderson_difference.c \
	$(ANDINC)anderson_difference.h $(ANDINC)anderson_reductions.h $(DENSE_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)anderson/
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

cpu_dense: cpu_vector cpu_matrix cpu_blas cpu_lapack cpu_dense_
gpu_dense: gpu_vector gpu_matrix gpu_blas gpu_dense_ #gpu_lapack

cpu_dense_: $(LAOUT)dense_cpu$(PRECISION).o
$(LAOUT)dense_cpu$(PRECISION).o: $(LASRC)dense.c $(LAINC)dense.h \
	$(LAINC)lapack.h $(LAINC)blas.h $(LAINC)matrix.h \
	$(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CC) $(CCFLAGS) $< -c -o $@

gpu_dense_: $(LAOUT)dense_gpu$(PRECISION).o
$(LAOUT)dense_gpu$(PRECISION).o: $(LASRC)dense.cu $(LAINC)dense.h \
	$(LAINC)lapack.h $(LAINC)blas.h $(LAINC)matrix.h $(LAINC)vector.h \
	$(GPU_DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CUXX) $(CUXXFLAGS) $< -c -o $@

cpu_lapack: $(LAOUT)lapack_cpu$(PRECISION).o
$(LAOUT)lapack_cpu$(PRECISION).o: $(LASRC)lapack.c $(LAINC)lapack.h \
	$(LAINC)matrix.h $(LAINC)vector.h $(DEF_HDR)
	mkdir -p $(OUT)
	mkdir -p $(OUT)linsys
	$(CC) $(CCFLAGS) $< -c -o $@

# gpu_lapack: $(LAOUT)lapack_gpu$(PRECISION).o
# $(LAOUT)lapack_gpu$(PRECISION).o: $(LASRC)lapack.cu $(LAINC)lapack.h \
# 	$(LAINC)matrix.h $(LAINC)vector.h $(GPU_DEF_HDR)
# 	mkdir -p $(OUT)
# 	mkdir -p $(OUT)linsys
# 	$(CUXX) $(CUXXFLAGS) $< -c -o $@

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