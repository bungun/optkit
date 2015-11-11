# Basic definitions
OPTKITROOT=./
OUT=$(OPTKITROOT)build/
SRC=$(OPTKITROOT)src/
PREFIX_OUT=$(OUT)optkit_

# C++ Flags
CXX=gcc
CXXFLAGS= -g -O3 -fPIC -I. -I./include -Wall -Wconversion
LDFLAGS=-lm


# Check system args
ifeq ($(shell uname -s), Darwin)
LDFLAGS+= -framework Accelerate
SHARED=dylib
else
LDFLAGS+= -lblas
SHARED=so
endif

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
ifneq ($(SPARSE), 0)
STATICTARG=gpu_sparse
else
STATICTARG=gpu_dense
endif
else
ifneq ($(SPARSE), 0)
STATICTARG=cpu_sparse
else
STATICTARG=cpu_dense
endif
endif
STATICLIB=$(PREFIX_OUT)$(STATICTARG)$(PRECISION).o



.PHONY: default, lib, cpu_dense 
#, cpu_sparse, gpu_dense, gpu_sparse
default: cpu_dense

libok: $(STATICTARG)
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o $(OUT)$@_$(STATICTARG)$(PRECISION).$(SHARED) $(STATICLIB) $(LDFLAGS)


cpu_dense: $(SRC)optkit_dense.c $(SRC)optkit_dense.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

gpu_dense: $(SRC)optkit_dense.cu $(SRC)optkit_dense.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

cpu_sparse: $(SRC)optkit_sparse.c $(SRC)optkit_sparse.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	

gpu_sparse: $(SRC)optkit_sparse.cu $(SRC)optkit_sparse.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $(PREFIX_OUT)$@$(PRECISION).o	



.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)