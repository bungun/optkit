import numpy as np
import optkit as ok

"""Nonnegative least squares.

Given problem data A (m-by-n matrix), b' (m-by-1 vector),
solve the problem

	min. ||Ax - b'||_2
	s.t. x >= 0.

In graph form, we phrase this as

	min. ||y - b'||^2 + I_{x' >= 0}(x)
	s.t. y = Ax.

To make it fully separable, we solve the equivalent problem

	min. ||y - b'||_2^2 + I_{x' >= 0}(x)
	s.t. y = Ax.

Thus, for POGS objective functions structured as:

	\phi(z) = \sum_{i=1}^m c_i * h_i(a_i * z_i - b_i) + d_i * z_i + e_i *z_i^2

we take:

	h_i = 'Square'
	a_i = 1 (default)
	b_i = b'_i
	c_i = 1 (default)
	d_i = 0 (default)
	e_i = 0 (default)

for the objective on y, and

	h_i = 'IndGe0'
	a_i = 1 (default)
	b_i = 0 (default)
	c_i = 1 (default)
	d_i = 0 (default)
	e_i = 0 (default)

for the objective on x.
"""

# PROBLEM DATA
n_variables = 500 + int(250 * np.random.rand())
n_constraints = 1500 + int(1000 * np.random.rand())

MEAN_VAL_B = 1.

A = np.random.rand(n_constraints, n_variables)
b = (2 * MEAN_VAL_B) * np.random.rand(n_constraints)

# SOLVER and OBJECTIVE FUNCTION SETUP
solver = ok.PogsSolver(A)

objecitve_constr = ok.PogsVariable(n_constraints, h='Square', b=b)
objective_vars = ok.PogsVariable(n_variables, h='IndGe0')

# FIRST SOLVE
solver.solve(verbose=2)
x_first = solver.output.x

# WARM START: PERTURB "b"
MEAN_PERTURB = MEAN_VAL_B / 10.
delta_b = 2 * MEAN_PERTURB * np.random.rand(n_constraints) - MEAN_PERTURB
b += delta_b

# update objective
objective_constr.set(b=b)

# setting flag resume=True warm starts solve based on solver state
solver.solve(objective_constr, objective_vars, verbose=2, resume=True)

x_second = solver.output.x
