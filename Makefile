# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
PREFIX_OUT=$(OUT)optkit_

# C++ Flags
CXX=gcc
CXXFLAGS= -g -O3 -fPIC -I. -I./include -Wall -Wconversion
LDFLAGS_=-lstdc++ -lm

# CUDA Flags
CUXX=nvcc
CUXXFLAGS=-arch=sm_50 -Xcompiler -fPIC
CULDFLAGS_=-lstdc++ -lm

# Check system args
ifeq ($(shell uname -s), Darwin)
LDFLAGS_+= -framework Accelerate
CULDFLAGS_+= -L/usr/local/cuda/lib 
SHARED=dylib
else
LDFLAGS_+= -lblas
CULDFLAGS_+= -L/usr/local/cuda/lib64 
SHARED=so
endif

CULDFLAGS_+= -lcudart -lcublas -lcusparse

# make options
GPU=0
SPARSE=0

# compiler options
FLOAT=0


# optional compiler flags 
OPT_FLAGS=
ifneq ($(FLOAT), 0)
OPT_FLAGS += -D=$(FLOAT) # use floats rather than doubles
endif

CXXFLAGS += $(OPT_FLAGS)


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

ifneq ($(SPARSE), 0)
MATRIXTAG=sparse
else
MATRIXTAG=dense
endif

STATICTARG=$(DEVICETAG)_$(MATRIXTAG)
PROXTARG=$(DEVICETAG)_prox

STATICLIB=$(PREFIX_OUT)$(STATICTARG)$(PRECISION).o
PROXLIB=$(PREFIX_OUT)$(PROXTARG)$(PRECISION).o


.PHONY: default, lib, cpu_dense 
#, cpu_sparse, gpu_dense, gpu_sparse
default: cpu_dense

libok: $(STATICTARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o $(OUT)$@_$(STATICTARG)$(PRECISION).$(SHARED) $(STATICLIB) $(LDFLAGS)

libprox: $(PROXTARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o $(OUT)$@_$(DEVICETAG)$(PRECISION).$(SHARED) $(PROXLIB) $(LDFLAGS)

cpu_dense: $(SRC)optkit_dense.c $(SRC)optkit_dense.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

gpu_dense: $(SRC)optkit_dense.cu $(SRC)optkit_dense.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

cpu_sparse: $(SRC)optkit_sparse.c $(SRC)optkit_sparse.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

gpu_sparse: $(SRC)optkit_sparse.cu $(SRC)optkit_sparse.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

cpu_prox: $(SRC)optkit_prox.c $(SRC)optkit_prox.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o

gpu_prox: $(SRC)optkit_prox.cu $(SRC)optkit_prox.h
	mkdir -p $(OUT)
	$(CUXX) $(CUXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o



.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)