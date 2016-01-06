	###v0.1 (targets)
- TODO: add test for vector_pow()
- TODO: OMP parallelization for C vector operations


###v0.0.2 (next release)
- TODO: error checking throughout GPU calls (and mirrored for CPU)
- TODO: automate testing with nose2
- TODO: license
- TODO: cite POGS
- TODO: documentation
- TODO: thorough commenting
- TODO: option to suppress [y, mu, nu] from solver output

###v0.0.1 (current)
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

