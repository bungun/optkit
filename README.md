# optkit

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

If desired, create a virtual environment before the installation step (https://virtualenv.pypa.io/en/latest/).

```bash
$ [sudo] PATH=$PATH python setup.py install
```


Usage
-----

After installing, import the package `optkit` to use it in a script, e.g.:

```python
>>>> import optkit as ok
```

The import should print a success messages:
```python

optkit backend set to cpu64
```

(TODO: further explanation)


Default Settings 
----------------

To change the default backend for the optkit module to attempt to bind upon 
import, set the following environment variables:

```bash
$ export OPTKIT\_DEFAULT\_DEVICE=<device>
```

(values = `cpu` or `gpu`; optkit defaults to CPU if environment variable 
not set), and 

```
$ export OPTKIT\_DEFAULT\_FLOATBITS=<bits>
```

(values = `64` or `32`; optkit defaults to 64-bit floating point precision if environment variable not set).


Credits
-------

optkit's dense linear algebra libraries wrap GNU CBlas and cudaBLAS.

Most of the linear algebra libraries, proximal operator libraries are C/CUDA adaptations of libraries implemented in C++/CUDA in POGS.

The projection, matrix equilibration, and POGS solver algorithms are Python adaptations of the corresponding C++ versions implemented in POGS.

**Visit http://foges.github.io/pogs/ for detailed information on POGS**

The following people have been, and are, involved in the development and maintenance of optkit
+ Baris Ungun (principal developer)
+ Stephen Boyd (methods and math)