###v0.1 (targets)
- TODO: OMP parallelization for C vector operations
- TODO: documentation
- TODO: automate testing with nose2


###v0.0.4 (next release)
- TODO: error checking throughout GPU calls (and mirrored for CPU)
- TODO: license
- TODO: cite POGS
- TODO: thorough commenting
- TODO: add unit test for vector_pow, _recip, _sqrt

###v0.0.3 (alpha)
- CPU Sparse Linear Algebra
- GPU Sparse Linear Algebra
- Python Sparse Linear Algebra bindings
- Separated FunctionVector (python implementation of POGS) from Objective (python wrapper for C implementation of POGS)
- Python backend counts allocations and calls cudaDeviceReset() when allocated blas handles and PogsSolver go out of scope (purpose: for scripts creating multiple PogsSolver instances, can delete solver objects to free GPU memory)
- Option to suppress [y, mu, nu] from POGS solver output (C and Py)

###v0.0.2 (current)
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
- block clustering in c/cuda
- adaptive alpha (line search) for POGS
- change Python bindings from ctypes -> Cython?
