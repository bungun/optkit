###v0.1 (targets)
- TODO: OMP parallelization for C vector operations
- TODO: documentation
- TODO: continuous integration
- TODO: Cython bindings
- TODO: operator typesafe equilibration
- TODO: operator IO

###v0.0.5 (next release)
- TODO: license
- TODO: cite POGS
- TODO: append version numbers to .so, check version numbers when loading libs in python
- TODO: add tests for vector_rand, transformable operators, operator export/import 

###v0.0.4 (beta)
- Migrate tests to unittests
- Eliminate slow Python implementations (to be replaced with Cython implementations in future, potentially)
- Abstract linear operators
- Operator implementations: dense, sparse, diagonal
- CGLS, preconditioned CG methods for abstract operators
- Pperator equilibration for: dense, sparse
- Approximate projection (with CGLS) for abstract linear operators
- Operator POGS for: dense, sparse
- Matrix, vector reductions (min, max, indmin); 
- Memory cleanup when tests fail 
- K-means clustering on rows of dense matrices
- Error checking throughout C libraries (all calls that don't return a pointer return an error code)

###v0.0.3 (current)
- CPU Sparse Linear Algebra
- GPU Sparse Linear Algebra
- Python Sparse Linear Algebra bindings
- Separate FunctionVector (python implementation of POGS) from Objective (python wrapper for C implementation of POGS)
- Python backend counts allocations and calls cudaDeviceReset() when allocated blas handles and PogsSolver go out of scope (purpose: for scripts creating multiple PogsSolver instances, can delete solver objects to free GPU memory)
- Option to suppress [y, mu, nu] from POGS solver output (C and Py)
- Version number calls in C, Py

###v0.0.2
- C implementations of equilibration, projection and POGS (dense only)

###v0.0.1
- CPU/GPU dense linear algebra (32/64-bit)
- CPU/GPU (separable) prox (32/64-bit)
- Sinkhorn-Knopp matrix equilibration (see http://arxiv.org/abs/1503.08366)
- Direct (dense) graph projection
- Dense, fully-separable implementation of POGS block splitting algorithm (see http://foges.github.io/pogs/ and http://arxiv.org/abs/1503.08366)
- I/O for saving solver state (equilbrated matrix, projector factorization, iterates)
- Backend switching

#ROADMAP:
###v0.2
- CPU EVD
- Python EVD bindings

###v0.2.1
- GPU EVD

###v0.3
- CPU cone prox
- CPU block prox
- Python cone prox bindings
- Python block prox bindings

###v0.3.1
- GPU cone prox
- GPU block prox

###v1.0
- Cone solver

###v++
- adaptive alpha (line search) for POGS
- DCP ADMM
- installation manifest (text or yaml) for clean uninstall
- operator l-BFGS preconditioner
- operator k-means