import collections

RHO_MAX = 1e4
RHO_MIN = 1e-4
DELTA_MAX = 2.
DELTA_MIN = 1.05
DELTA_MOD = 1.01
TOL_CONTRACTION = 0.9
DELAY = 1.25

RhoParams = collections.namedtuple(
        'AdaptiveRhoParameters',
        'rho_max, rho_min, delta_max, delta_min, delta_mod, tol_contraction, delay')

class AdaptiveRho:
    def __init__(self, **options):
        self.delta = DELTA_MIN
        self.last_rho_increase = 0
        self.last_rho_decrease = 0
        self.tol_scaling = 1.
        self.params = RhoParams(
                options.pop('rho_max', RHO_MAX),
                options.pop('rho_min', RHO_MIN),
                options.pop('delta_max', DELTA_MAX),
                options.pop('delta_min', DELTA_MIN),
                options.pop('delta_mod', DELTA_MOD),
                options.pop('tol_contraction', TOL_CONTRACTION),
                options.pop('delay', DELAY))

    def __call__(self, rho, dual_iterate, primal_residual, dual_residual,
                 primal_tolerance, dual_tolerance, iteration, verbose=False):
        (RHO_MAX, RHO_MIN, DELTA_MAX, DELTA_MIN, DELTA_MOD, TOL_CONTRACTION,
            DELAY) = self.params
        if (
                dual_residual < self.tol_scaling * dual_tolerance and
                primal_residual > self.tol_scaling * primal_tolerance and
                iteration > delay * self.last_rho_decrease):
            if rho < rho_max:
                rho *= self.delta
                dual_iterate /= self.delta
                self.delta = min(self.delta * DELTA_MOD, DELTA_MAX)
                self.last_rho_increase = iteration
                if verbose:
                    print('+RHO: {:3e}\n'.format(rho))
        elif (
                dual_residual > self.tol_scaling * dual_tolerance and
                primal_residual < self.tol_scaling * primal_tolerance and
                iteration > DELAY * self.last_rho_increase):
            if rho > RHO_MIN:
                rho /= self.delta
                dual_iterate *= self.delta
                self.delta = min(self.delta * DELTA_MOD, DELTA_MAX)
                self.last_rho_decrease = iteration
                if verbose:
                    print('-RHO: {:3e}\n'.format(rho))
        elif (
                dual_residual < self.tol_scaling * dual_tolerance and
                primal_residual < self.tol_scaling * primal_tolerance):
            self.tol_scaling *= TOL_CONTRACTION
        else:
            self.delta = max(self.delta / DELTA_MOD, DELTA_MIN)

        return rho, dual_iterate

