# optkit

[![Build Status](https://travis-ci.org/bungun/optkit.svg?branch=master)](https://travis-ci.org/bungun/optkit)


## About `optkit`

This library provides:
<!-- + a unified Python interface for (standard) CPU and GPU BLAS dense linear algebra libraries -->
<!-- + a unified Python interface for (custom) CPU and GPU proximal operator libraries -->
<!-- + Python implementations of projection and matrix equilibration routines  -->
<!-- + a Python implementation of the ADMM solver POGS, based on Chris Fougner's convex optimization solver library (http:/github.com/foges/pogs). -->
	 
Requirements
------------
optkit's C/CUDA backend libraries have the following dependencies:

	cBLAS
	CUDA >= 7.5

optkit's Python package has the following additional dependencies:

	python >= 2.7
	numpy >= 1.8
	scipy >= 0.13
	toolz	


Installation
------------

###<u> Building from source

To download `optkit` from GitHub: 

```bash
git clone https://github.com/bungun/python-pogs --recursive
```

Alternately, you can execute
```bash
git clone https://github.com/bungun/optkit
```

If would like to compile POGS for GPU, please make sure your `$PATH` environment variable contains the path to CUDA binaries (namely `nvcc`) and libraries.


```bash
$ export PATH=$PATH:<path-to-cuda-bin>
$ export LD\_LIBRARY\_PATH=$LD\_LIBRARY\_PATH:<path-to-cuda-libs> 
```

(For instance, on Linux distributions, these paths would be `/usr/local/cuda/bin` and `/usr/local/cuda/lib64`, respectively.)


In the shell, navigate to the directory:

```bash
$ cd <path-to-optkit>/python
```

(Optional, recommended.) If desired, create a virtual environment before the installation step (e.g., see https://virtualenv.pypa.io/en/latest/ or http://conda.pydata.org/docs/using/envs.html#create-an-environment). For instance:

```bash
$ conda create -n <optkit-environment-name> numpy scipy toolz nose
$ source activate <optkit-environment-name>
``` 

Finally, to install:

```bash
$ python setup.py install
```

(If installing as root with `sudo`, the contents of `PATH` need to be passed
through

```bash
$ sudo PATH=$PATH python setup.py install
```
for the install to succeed. This is not needed when using a virtual environment.)


Usage
-----

After installing, import the package `optkit` to use it in a script, e.g.:

```python
> import optkit as ok
```

The import should print a success message:
```python

"optkit backend set to cpu64"
```

To change the backend, call:
```python
> ok.set_backend(double=True, gpu=False)
```

(TODO: further explanation)


Default Settings 
----------------

To change the default backend for the optkit module to attempt to bind upon 
import, set the following environment variables:

```bash
$ export OPTKIT\_DEFAULT\_DEVICE=<device>
```

(values: `cpu` or `gpu`; optkit defaults to CPU if environment variable 
not set), and 

```bash
$ export OPTKIT\_DEFAULT\_FLOATBITS=<bits>
```

(values: `64` or `32`; optkit defaults to 64-bit floating point precision if environment variable not set).


Credits
-------

`optkit`'s dense linear algebra libraries wrap GNU CBlas and cudaBLAS.

Most of the linear algebra libraries, proximal operator libraries are C/CUDA adaptations of libraries implemented in C++/CUDA in POGS.

The conjugate gradient-least squares, projection, matrix equilibration, and POGS solver algorithms are C adaptations of the corresponding C++ versions implemented in POGS.

+ **Visit http://foges.github.io/pogs/ for detailed information on POGS**

The preconditioned conguate gradient algorithm is adapted from the implementation
in SCS.

+ **Visit https://github.com/cvxgrp/scs for detailed information on SCS**

The following people have been, and are, involved in the development and maintenance of optkit
+ Baris Ungun (principal developer)
+ Chris Fougner (POGS, CGLS, proximal operators)
+ Stephen Boyd (methods and math)