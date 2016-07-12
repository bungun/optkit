import numpy as np
import optkit as ok

"""Radiation treatment planning.

For a radiation treatment planning case with m voxels and n candidate
beams, we require a dose-influence matrix A (m-by-n), and the
desired/prescribed dose vector d' (m-by-1), and set up the problem
solve the problem

	min. \sum_{i=1}^m c_i|y_i - d'_i| + d_i(y_i - d'_i)
	s.t.  y = Ax
		  x >= 0.

This formulation applies a simple piecewise linear objective to the
achieved voxel doses, with the parameters c_i and d_i combining to
control the slope of underdose and overdose penalties, i.e.:

penalty on voxel i =	(c_i + rx_i) (y_i - d'_i), 		y_i >= d'_i
						(-c_i + rx_i)(y_i - d'_i), 		y_i <  d'_i.


This problem is already in graph form.

Thus, for POGS objective functions structured as:

	\phi(z) = \sum_{i=1}^m c_i * h_i(a_i * z_i - b_i) + d_i * z_i + e_i *z_i^2

we take:

	h_i = 'Abs'
	a_i = 1 (default)
	b_i = dose_i
	c_i = 1 (default)
	d_i = 0 (default)
	e_i = 0 (default)

for the objective on y (which is offset by a constant -d^Td' from the
desired objective), and

	h_i = 'IndGe0'
	a_i = 1 (default)
	b_i = 0 (default)
	c_i = 1 (default)
	d_i = 0 (default)
	e_i = 0 (default)

for the objective on x.
"""

# normalized prescription
rx = 1.

# problem dimensions
n_target = 5000
n_organ1 = 12000
n_organ2 = 3000
n_voxels = n_target + n_organ1 + n_organ2
n_beams = 2000

# generate random dose matrices, with voxels in target getting ~5x
# the average dose of non-target voxels, say.
A_target = np.random.rand(n_target, n_beams)
A_organ1 = 0.2 * np.random.rand(n_organ1, n_beams)
A_organ2 = 0.2 * np.random.rand(n_organ2, n_beams)

A = np.vstack((A_target, A_organ1, A_organ2))

# prescribed dose = <rx> for target, 0 elsewhere
dose = np.zeros(n_voxels)
dose[:n_target] = rx

# objective weights:
overdose_target = 1. / float(n_target)
underdose_target = 0.05 / float(n_target)
overdose_organ1 = 0.033 / float(n_organ1)
overdose_organ2 = 0.033 / float(n_organ2)

weight_abs = np.zeros(n_voxels)
weight_abs[:n_target] = (overdose_target + underdose_target) / 2.
weight_abs[n_target:n_target + n_organ1] = overdose_organ1
weight_abs[-n_organ2:] = overdose_organ2

weight_linear = np.zeros(n_voxels)
weight_linear[:n_target] = (overdose_target - underdose_target) / 2.

# POGS objectives
objective_voxels = ok.PogsObjective(n_voxels, h='Abs', b=dose, c=weight_abs,
									d=weight_linear)
objective_beams = ok.PogsObjective(n_beams, h='IndGe0')

# plan case
solver = ok.PogsSolver(A)
solver.solve(objective_voxels, objective_beams)

# retrieve output:
# 	x >= 0 enforced exactly, so beam weights will be feasible
#	Ax = y projected to solver tolerance, so voxel doses may be slightly
#			better than what the beam weights achieve
optimal_beam_weights = solver.output.x
optimal_voxel_doses = solver.output.y

# calculate actual voxel doses given optimal beam weights
acheived_voxel_doses = A.dot(optimal_beam_weights)

# print some stats
mean_target = sum(acheived_voxel_doses[:n_target])/float(n_target)
mean_1 = sum(acheived_voxel_doses[n_target:n_target + n_organ1])/float(n_organ1)
mean_2 = sum(acheived_voxel_doses[-n_organ2:])/float(n_organ2)

print('MEAN DOSE, TARGET: {}'.format(mean_target))
print('MEAN DOSE, ORGAN 1: {}'.format(mean_1))
print('MEAN DOSE, ORGAN 2: {}'.format(mean_2))


