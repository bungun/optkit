#FEATURES:
###v0.0.0
- CPU/GPU dense linear algebra (32/64-bit)
- CPU/GPU (separable) prox (32/64-bit)
- Sinkhorn-Knopp matrix equilibration (see http://arxiv.org/abs/1503.08366)
- Dense/Direct graph projection
- Dense, fully-separable implementation of POGS block splitting algorithm (see http://foges.github.io/pogs/ and http://arxiv.org/abs/1503.08366)


#TODO:
###v0.0.1 (next release)
- error checking throughout GPU calls (and mirrored for CPU)
- 32/64-bit flags & checks in python bindings
- test/finalize setup.py
- automate testing with nose2
- license
- cite POGS

###v0.0.2 (targets)
- Back-end encapsulation
- Back-end switching

#ROADMAP:
###v0.1
- GPU vector_pow()
- add test for vector_pow()
- OMP parallelization for C vector operations
- move equilibration from Python to C[/CUDA]

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

