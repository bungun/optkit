import numpy as np
import optkit as ok

def intensity_optimization(shape, target_dose=1., percent_target=30,
                           **options):
    r"""Generate a random radiation treatment planning problem.

    For a radiation treatment planning case with m voxels and n candidate
    beams, we require a nonnegative dose-influence matrix A (m-by-n),
    and the desired/prescribed dose vector d' (m-by-1), and set up the
    problem solve the problem:

        min. \sum_{i=1}^m c_i|y_i - d'_i| + d_i(y_i - d'_i)
        s.t.  y = Ax
              x >= 0.

    This formulation applies a simple piecewise linear objective to the
    achieved voxel doses, with the parameters c_i and d_i combining to
    control the slope of underdose and overdose penalties, i.e.:

    penalty on voxel i =    (c_i + rx_i) (y_i - d'_i),      y_i >= d'_i
                            (-c_i + rx_i)(y_i - d'_i),      y_i <  d'_i.


    This problem is already in graph form.
    """
    m, n = n_voxels, n_beams = shape
    size_target = int(m * max(0, min(100, float(percent_target))) / 100.)
    size_nontarget = m - size_target

    # generate A random dose matrix, with voxels in target getting ~5x
    # the average dose of non-target voxels.
    A = np.random.random((m, n))
    A[size_target:, :] *= 0.2

    # prescribed dose = `target_dose` for target, 0 elsewhere
    dose = float(target_dose)

    # objective weights:
    overdose_target = 1.
    underdose_target = 0.05
    overdose_nontarget = 0.033 * (float(size_target) / float(size_nontarget))

    weight_abs = np.zeros(n_voxels)
    weight_abs[:size_target] = (overdose_target + underdose_target) / 2.

    weight_linear = np.zeros(n_voxels)
    weight_linear[:size_target] = (overdose_target - underdose_target) / 2.
    weight_linear[size_target:] = overdose_nontarget

    # POGS objectives
    objective_voxels = ok.PogsObjective(n_voxels, h='Abs', b=dose, c=weight_abs,
                                        d=weight_linear)
    objective_beams = ok.PogsObjective(n_beams, h='IndGe0')

    return A, objective_voxels, objective_beams
