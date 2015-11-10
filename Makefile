# C++ Flags
CXX=gcc
CXXFLAGS= -g -O3 -fPIC -I. -I./include -Wall -Wconversion
LDFLAGS=-lm


# Check System Args.
ifeq ($(shell uname -s), Darwin)
LDFLAGS+= -framework Accelerate
SHARED=dylib
else
LDFLAGS+= -lblas
SHARED=so
endif

OPT_FLAGS=
# optional compiler flags 
FLOAT = 0
ifneq ($(FLOAT), 0)
OPT_FLAGS += -D=$(FLOAT) # use floats rather than doubles
endif

CXXFLAGS += $(OPT_FLAGS)

GSLROOT=./
OUT=$(GSLROOT)build/


.PHONY: default, libgsl, gsl
default: gsl

libgsl: $(OUT)/gsl.o
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -shared -o $(OUT)$@.$(SHARED) $< $(LDFLAGS)

gsl: $(OUT)/gsl.o

$(OUT)/gsl.o: gsl.c gsl.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) $< -c -o $@ $(LDFLAGS)	


.PHONY: clean
clean:
	rm -f *.o *.so *.dylib *~ *~ 
	rm -rf *.dSYM
	rm -rf $(OUT)