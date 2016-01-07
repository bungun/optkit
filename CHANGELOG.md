	###v0.1 (targets)
- TODO: OMP parallelization for C vector operations
- TODO: documentation
- TODO: automate testing with nose2


###v0.0.3 (next release)
- TODO: error checking throughout GPU calls (and mirrored for CPU)
- TODO: license
- TODO: cite POGS
- TODO: thorough commenting
- TODO: option to suppress [y, mu, nu] from solver output
- TODO: add unit test for vector_pow, _recip, _sqrt

###v0.0.2 (current)
- C implementations of equilibration, projection and POGS

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
- CPU Sparse Linear Algebra
- Python Sparse Linear Algebra bindings

###v0.2.1
- GPU Sparse Linear Algebra

###v0.3
- CPU EVD
- Python EVD bindings

###v0.3.1
- GPU EVD

###v0.4
- CPU cone prox
- Python cone prox bindings

###v0.4.1
- GPU cone prox

###v1.0
- Cone solver

###v++
- block clustering in c/cuda
- adaptive alpha
- change Python bindings from ctypes -> Cython?
