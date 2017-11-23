from optkit.compat import *

import os
import numpy as np

from optkit.utils import proxutils
from optkit.tests.defs import TEST_ITERATE
from optkit.tests.C.base import OptkitCTestCase

ALPHA_DEFAULT = 1.7
RHO_DEFAULT = 1
MAXITER_DEFAULT = 2000
ABSTOL_DEFAULT = 1e-4
RELTOL_DEFAULT = 1e-3
ADAPTIVE_DEFAULT = 1
GAPSTOP_DEFAULT = 0
WARMSTART_DEFAULT = 0
VERBOSE_DEFAULT = 2
SUPPRESS_DEFAULT = 0
RESUME_DEFAULT = 0

class PogsBaseTestContext(object):
    def __init__(self, lib, m, n, obj='Abs'):
        self._lib = lib
        self._f = oktest.CFunctionVectorContext(lib, m)
        self._g = oktest.CFunctionVectorContext(lib, n)
        self._solver = solver
        self.params = collections.namedtuple(
                'TestParams',
                'shape z output info setings res tol obj f f_py g g_py')
        m, n = self.params.shape = self.A.py.shape
        self.params.z = self.PogsVariablesLocal(lib, m, n)
        self.params.z = self.PogsVariablesLocal(m, n, lib.pyfloat)
        self.params.output = self.PogsOutputLocal(lib, m, n)
        self.params.info = lib.pogs_info()
        self.params.settings = lib.pogs_settings()
        oktest.assert_noerr( lib.set_default_settings(settings) )
        self.params.settings.verbose = int(self.VERBOSE_TEST)
        self.params.res = lib.pogs_residuals()
        self.params.tol = lib.pogs_tolerances()
        self.params.obj = lib.pogs_objective_values()

    def __enter__(self):
        self.lib = self._lib.__enter__()
        self.f = self._f.__enter__()
        self.g = self._g.__enter__()
        f, f_py, f_ptr = self.f.cptr, self.f.py, self.f.pyptr
        g, g_py, g_ptr = self.g.cptr, self.g.py, self.g.pyptr

        h = lib.function_enums.dict[objective]
        asymm = 1 + int('Asymm' in objective)
        for i in xrange(m):
            f_py[i] = self.lib.function(h, 1, 1, 1, 0, 0, asymm)
        for j in xrange(n):
            g_py[j] = self.lib.function(
                    lib.function_enums.IndGe0, 1, 0, 1, 0, 0, 1)
        oktest.assert_noerr( self.lib.function_vector_memcpy_va(f, f_ptr) )
        oktest.assert_noerr( self.lib.function_vector_memcpy_va(g, g_ptr) )
        self.params.f = f
        self.params.g = g
        self.params.f_py = f_py
        self.params.g_py = g_py
        return self

class OptkitCPogsTestCase(OptkitCTestCase):
    class PogsVariablesLocal():
        def __init__(self, m, n, pytype):
            self.m = m
            self.n = n
            self.z = np.zeros(m + n).astype(pytype)
            self.z12 = np.zeros(m + n).astype(pytype)
            self.zt = np.zeros(m + n).astype(pytype)
            self.zt12 = np.zeros(m + n).astype(pytype)
            self.prev = np.zeros(m + n).astype(pytype)
            self.d = np.zeros(m).astype(pytype)
            self.e = np.zeros(n).astype(pytype)

        @property
        def x(self):
            return self.z[self.m:]

        @property
        def y(self):
            return self.z[:self.m]

        @property
        def x12(self):
            return self.z12[self.m:]

        @property
        def y12(self):
            return self.z12[:self.m]

        @property
        def xt(self):
            return self.zt[self.m:]

        @property
        def yt(self):
            return self.zt[:self.m]

        @property
        def xt12(self):
            return self.zt12[self.m:]

        @property
        def yt12(self):
            return self.zt12[:self.m]

    class PogsOutputLocal():
        def __init__(self, lib, m, n):
            self.x = np.zeros(n).astype(lib.pyfloat)
            self.y = np.zeros(m).astype(lib.pyfloat)
            self.mu = np.zeros(n).astype(lib.pyfloat)
            self.nu = np.zeros(m).astype(lib.pyfloat)
            self.ptr = lib.pogs_output(self.x.ctypes.data_as(lib.ok_float_p),
                                       self.y.ctypes.data_as(lib.ok_float_p),
                                       self.mu.ctypes.data_as(lib.ok_float_p),
                                       self.nu.ctypes.data_as(lib.ok_float_p))

    def load_to_local(self, lib, py_vector, c_vector):
        self.assertCall( lib.vector_memcpy_av(
                py_vector.ctypes.data_as(lib.ok_float_p), c_vector, 1) )

    def load_all_local(self, lib, py_vars, solver):
        if not isinstance(py_vars, self.PogsVariablesLocal):
            raise TypeError('argument "py_vars" must be of type {}'.format(
                            self.PogsVariablesLocal))
        elif 'pogs_solver_p' not in lib.__dict__:
            raise ValueError('argument "lib" must contain field named '
                             '"pogs_solver_p"')
        elif not isinstance(solver, lib.pogs_solver_p):
            raise TypeError('argument "solver" must be of type {}'.format(
                            lib.pogs_solver_p))

        z = solver.contents.z
        self.load_to_local(lib, py_vars.z, z.contents.primal.contents.vec)
        self.load_to_local(lib, py_vars.z12, z.contents.primal12.contents.vec)
        self.load_to_local(lib, py_vars.zt, z.contents.dual.contents.vec)
        self.load_to_local(lib, py_vars.zt12, z.contents.dual12.contents.vec)
        self.load_to_local(lib, py_vars.prev, z.contents.prev.contents.vec)

        if 'pogs_matrix_p' in lib.__dict__:
            solver_work = solver.contents.M
        elif 'pogs_work_p' in lib.__dict__:
            solver_work = solver.contents.W
        else:
            raise ValueError('argument "lib" must contain a field named'
                             '"pogs_matrix_p" or "pogs_work_p"')

        self.load_to_local(lib, py_vars.d, solver_work.contents.d)
        self.load_to_local(lib, py_vars.e, solver_work.contents.e)

    class EquilibratedMatrix(object):
        def __init__(self, d, A, e):
            self.dot = lambda x: d * A.dot(e * x)
            self.shape = A.shape
            self.transpose = lambda: EquilibratedMatrix(e, A.T, d)
        @property
        def T(self):
            return self.transpose()


    def assert_default_settings(self, lib):
        settings = lib.pogs_settings()

        TOL = 1e-3
        self.assertCall( lib.set_default_settings(settings) )
        self.assertScalarEqual(settings.alpha, ALPHA_DEFAULT, TOL )
        self.assertScalarEqual(settings.rho, RHO_DEFAULT, TOL )
        self.assertScalarEqual(settings.abstol, ABSTOL_DEFAULT, TOL )
        self.assertScalarEqual(settings.reltol, RELTOL_DEFAULT, TOL )
        self.assertScalarEqual(settings.maxiter, MAXITER_DEFAULT,
                                     TOL )
        self.assertScalarEqual(settings.verbose, VERBOSE_DEFAULT,
                                     TOL )
        self.assertScalarEqual(settings.suppress, SUPPRESS_DEFAULT,
                                     TOL )
        self.assertScalarEqual(settings.adaptiverho,
                                     ADAPTIVE_DEFAULT, TOL )
        self.assertScalarEqual(settings.gapstop, GAPSTOP_DEFAULT, TOL )
        self.assertScalarEqual(settings.warmstart, WARMSTART_DEFAULT,
                                     TOL )
        self.assertScalarEqual(settings.resume, RESUME_DEFAULT, TOL )

    def assert_pogs_scaling(self, lib, solver, test_params):
        f, f_py = test_params.f, test_params.f_py
        g, g_py = test_params.g, test_params.g_py
        local_vars = test_params.z
        m, n = test_params.shape

        def fv_list2arrays(function_vector_list):
            fv = function_vector_list
            fvh = np.array([fv_.h for fv_ in fv])
            fva = np.array([fv_.a for fv_ in fv])
            fvb = np.array([fv_.b for fv_ in fv])
            fvc = np.array([fv_.c for fv_ in fv])
            fvd = np.array([fv_.d for fv_ in fv])
            fve = np.array([fv_.e for fv_ in fv])
            return fvh, fva, fvb, fvc, fvd, fve

        # record original function vector parameters
        f_list = [lib.function(*f_) for f_ in f_py]
        g_list = [lib.function(*f_) for f_ in g_py]
        f_h0, f_a0, f_b0, f_c0, f_d0, f_e0 = fv_list2arrays(f_list)
        g_h0, g_a0, g_b0, g_c0, g_d0, g_e0 = fv_list2arrays(g_list)

        # copy function vector
        f_py_ptr = f_py.ctypes.data_as(lib.function_p)
        g_py_ptr = g_py.ctypes.data_as(lib.function_p)
        self.assertCall( lib.function_vector_memcpy_va(f, f_py_ptr) )
        self.assertCall( lib.function_vector_memcpy_va(g, g_py_ptr) )

        # scale function vector
        self.assertCall( lib.pogs_scale_objectives(
                solver.contents.f, solver.contents.g, f, g,
                solver.contents.A.contents.d,
                solver.contents.A.contents.e) )

        # retrieve scaled function vector parameters
        self.assertCall( lib.function_vector_memcpy_av(f_py_ptr,
                                                       solver.contents.f) )
        self.assertCall( lib.function_vector_memcpy_av(g_py_ptr,
                                                       solver.contents.g) )
        f_list = [lib.function(*f_) for f_ in f_py]
        g_list = [lib.function(*f_) for f_ in g_py]
        f_h1, f_a1, f_b1, f_c1, f_d1, f_e1 = fv_list2arrays(f_list)
        g_h1, g_a1, g_b1, g_c1, g_d1, g_e1 = fv_list2arrays(g_list)

        # retrieve scaling
        self.load_all_local(lib, local_vars, solver)

        # scaled vars
        self.assertVecEqual( f_a0, local_vars.d * f_a1, self.ATOLM, self.RTOL )
        self.assertVecEqual( f_d0, local_vars.d * f_d1, self.ATOLM, self.RTOL )
        self.assertVecEqual( f_e0, local_vars.d * f_e1, self.ATOLM, self.RTOL )
        self.assertVecEqual( g_a0 * local_vars.e, g_a1, self.ATOLN, self.RTOL )
        self.assertVecEqual( g_d0 * local_vars.e, g_d1, self.ATOLN, self.RTOL )
        self.assertVecEqual( g_e0 * local_vars.e, g_e1, self.ATOLN, self.RTOL )

        # unchanged vars
        self.assertVecEqual( f_h0, f_h1, self.ATOLM, self.RTOL )
        self.assertVecEqual( f_b0, f_b1, self.ATOLM, self.RTOL )
        self.assertVecEqual( f_c0, f_c1, self.ATOLM, self.RTOL )
        self.assertVecEqual( g_h0, g_h1, self.ATOLN, self.RTOL )
        self.assertVecEqual( g_b0, g_b1, self.ATOLN, self.RTOL )
        self.assertVecEqual( g_c0, g_c1, self.ATOLN, self.RTOL )

    def build_A_equil(self, lib, solver_work, A_orig):
        m, n = A_orig.shape
        d_local = np.zeros(m).astype(lib.pyfloat)
        e_local = np.zeros(n).astype(lib.pyfloat)
        self.load_to_local(lib, d_local, solver_work.contents.d)
        self.load_to_local(lib, e_local, solver_work.contents.e)
        return EquilibratedMatrix(d_local, A_orig, e_local,)

    def assert_equilibrate_matrix(lib, solver_work, A_orig):
        m, n = A_orig.shape
        d_local = np.zeros(m).astype(lib.pyfloat)
        e_local = np.zeros(n).astype(lib.pyfloat)
        self.load_to_local(lib, d_local, solver_work.contents.d)
        self.load_to_local(lib, e_local, solver_work.contents.e)

        x, x_py, x_ptr = self.register_vector(lib, n, 'x', random=True)
        y, y_py, y_ptr = self.register_vector(lib, m, 'y')

        if lib.py_pogs_impl == 'dense':
            A = solver_work.contents.A
            hdl = solver_work.contents.linalg_handle
            order = A.contents.order
            self.assertCall( lib.blas_gemv(hdl, order, alpha, 1, A, x, 0, y) )
        elif lib.py_pogs_impl == 'abstract':
            opA = solver_work.contents.A
            self.assertCall( opA.contents.apply(opA.contents.data, x, y) )
        else:
            raise ValueError('UNKNOWN POGS IMPLEMENTATION')

        self.load_to_local(lib, y_py, y)

        DAEX = d * A_orig.dot(e * x_py)
        self.assertVecEqual( y_py, DAEX, self.ATOLM, self.RTOL )
        self.free_vars('x', 'y')

    def assert_apply_matrix(self, lib, work, A_equil):
        m, n = A_equil.shape
        x, x_py, _ = self.register_vector(lib, n, 'x', random=True)
        y, y_py, _ = self.register_vector(lib, m, 'y', random=True)
        alpha = np.random.random()
        beta = np.random.random()

        if lib.py_pogs_impl == 'dense':
            self.assertCall( lib.pogs_dense_apply_matrix(
                    work, alpha, x, beta, y) )
        elif lib.py_pogs_impl == 'abstract':
            self.assertCall( lib.pogs_abstract_apply_matrix(
                    work, alpha, x, beta, y) )
        else:
            raise ValueError('UNKNOWN POGS IMPLEMENTATION')

        AXPY = alpha * A_equil.dot(x_py) + beta * y_py
        self.load_to_local(lib, y_py, y)
        self.assertVecEqual( y_py, AXPY, self.ATOLM, self.RTOL )

        self.free_vars('x', 'y')

    def assert_apply_adjoint(self, lib, work, A_equil):
        m, n = A_equil.shape
        x, x_py, _ = self.register_vector(lib, n, 'x', random=True)
        y, y_py, _ = self.register_vector(lib, m, 'y', random=True)
        alpha = np.random.random()
        beta = np.random.random()

        if lib.py_pogs_impl == 'dense':
            self.assertCall( lib.pogs_dense_apply_adjoint(
                    work, alpha, y, beta, x) )
        elif lib.py_pogs_impl == 'abstract':
            self.assertCall( lib.pogs_abstract_apply_adjoint(
                    work, alpha, y, beta, x) )
        else:
            raise ValueError('UNKNOWN POGS IMPLEMENTATION')

        ATYPX = alpha * A_equil(y_py) + beta * x_py
        self.load_to_local(lib, x_py, x)
        self.assertVecEqual( x_py, ATYPX, self.ATOLN, self.RTOL )

        self.free_vars('x', 'y')

    def assert_project_graph(self, lib, work, A_equil):
        m, n = A_equil.shape
        DIGITS = 7 - 2 * lib.FLOAT
        RTOL = 10**(-DIGITS)
        ATOLM = RTOL * m**0.5

        x_in, _, _ = self.register_vector(lib, n, 'x_in', random=True)
        y_in, _, _ = self.register_vector(lib, m, 'y_in', random=True)
        x_out, x_out_py, _ = self.register_vector(lib, n, 'x_out')
        y_out, y_out_py, _ = self.register_vector(lib, m, 'y_out')

        if lib.py_pogs_impl == 'dense':
            self.assertCall( lib.pogs_dense_project_graph(
                    work, x_in, y_in, x_out, y_out, RTOL) )
        elif lib.py_pogs_impl == 'abstract':
            self.assertCall( lib.pogs_abstract_project_graph(
                    work, x_in, y_in, x_out, y_out, RTOL) )
        else:
            raise ValueError('UNKNOWN POGS IMPLEMENTATION')

        self.load_to_local(lib, x_out_py, x_out)
        self.load_to_local(lib, y_out_py, y_out)
        AX = A_equil.dot(x_out_py)
        self.assertVecEqual( y_py, AX, ATOLM, RTOL )

        self.free_vars('x_in', 'y_in', 'x_out', 'y_out')

    def assert_pogs_primal_update(self, lib, solver, local_vars):
        """primal update test

            set

                z^k = z^{k-1}

            check

                z^k == z^{k-1}

            holds elementwise
        """
        self.assertCall( lib.pogs_primal_update(solver.contents.z) )
        self.load_all_local(lib, local_vars, solver)
        self.assertVecEqual( local_vars.z, local_vars.prev, self.ATOLMN, self.RTOL )

    def assert_pogs_prox(self, lib, solver, test_params):
        """proximal operator application test

            set

                x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
                y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

            in C and Python, check that results agree
        """
        f, f_py = test_params.f, test_params.f_py
        g, g_py = test_params.g, test_params.g_py
        local_vars = test_params.z

        z = solver.contents.z
        rho = solver.contents.rho
        self.assertCall( lib.pogs_prox(solver.contents.blas_handle, f, g, z, rho) )
        self.load_all_local(lib, local_vars, solver)

        f_list = [lib.function(*f_) for f_ in f_py]
        g_list = [lib.function(*f_) for f_ in g_py]
        for i in xrange(len(f_py)):
            f_list[i].a *= local_vars.d[i]
            f_list[i].d *= local_vars.d[i]
            f_list[i].e *= local_vars.d[i]
        for j in xrange(len(g_py)):
            g_list[j].a /= local_vars.e[j]
            g_list[j].d /= local_vars.e[j]
            g_list[j].e /= local_vars.e[j]

        x_arg = local_vars.x - local_vars.xt
        y_arg = local_vars.y - local_vars.yt
        x_out = proxutils.prox_eval_python(g_list, rho, x_arg)
        y_out = proxutils.prox_eval_python(f_list, rho, y_arg)
        self.assertVecEqual( local_vars.x12, x_out, self.ATOLN, self.RTOL )
        self.assertVecEqual( local_vars.y12, y_out, self.ATOLM, self.RTOL )

    def assert_pogs_project_graph(self, lib, solver, A_equil, local_vars):
        """primal projection test
            set
                (x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})
            check that
                y^{k+1} == A * x^{k+1}
            holds to numerical tolerance
        """
        self.assertCall( lib.pogs_project_graph(
                solver.contents.W, solver.contents.z,
                solver.contents.settings.contents.alpha, self.RTOL) )

        self.load_all_local(lib, local_vars, solver)
        self.assertVecEqual(A_equil.dot(x), local_vars.y, self.ATOLM, self.RTOL)

    def assert_pogs_dual_update(self, lib, solver, local_vars):
        """dual update test

            set

                zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
                zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
                                              (1-alpha) * z^k

            in C and Python, check that results agree
        """
        blas_handle = solver.contents.blas_handle
        z = solver.contents.z

        self.load_all_local(lib, local_vars, solver)
        alpha = solver.contents.settings.contents.alpha
        zt12_py = local_vars.z12 - local_vars.prev + local_vars.zt
        zt_py = local_vars.zt - local_vars.z + (
                    alpha * local_vars.z12 + (1-alpha) * local_vars.prev)

        self.assertCall( lib.pogs_dual_update(blas_handle, z, alpha) )
        self.load_all_local(lib, local_vars, solver)
        self.assertVecEqual( local_vars.zt12, zt12_py, self.ATOLMN, self.RTOL )
        self.assertVecEqual( local_vars.zt, zt_py, self.ATOLMN, self.RTOL )


    def assert_pogs_adapt_rho(self, lib, solver, test_params):
        """adaptive rho test

            rho is rescaled
            dual variable is rescaled accordingly

            the following equality should hold elementwise:

                rho_before * zt_before == rho_after * zt_after

            (here zt, or z tilde, is the dual variable)
        """
        residuals = test_params.res
        tolerances = test_params.tol
        local_vars = test_params.z

        rho_p = lib.ok_float_p()
        rho_p.contents = lib.ok_float(solver.contents.rho)

        z = solver.contents.z
        settings = solver.contents.settings

        if settings.contents.adaptiverho:
            rho_params = lib.adapt_params(1.05, 0, 0, 1)
            zt_before = local_vars.zt
            rho_before = solver.contents.rho
            self.assertCall( lib.pogs_adapt_rho(
                    z, rho_p, rho_params, solver.contents.settings, residuals,
                    tolerances, 1) )
            self.load_all_local(lib, local_vars, solver)
            zt_after = local_vars.zt
            rho_after = rho_p.contents
            self.assertVecEqual(
                    rho_after * zt_after, rho_before * zt_before, self.ATOLMN,
                    self.RTOL )

    def assert_pogs_check_convergence(self, lib, solver, A_equil, test_params):
        """convergence test

            (1) set

                obj_primal = f(y^{k+1/2}) + g(x^{k+1/2})
                obj_gap = <z^{k+1/2}, zt^{k+1/2}>
                obj_dual = obj_primal - obj_gap

                tol_primal = abstol * sqrt(m) + reltol * ||y^{k+1/2}||
                tol_dual = abstol * sqrt(n) + reltol * ||xt^{k+1/2}||

                res_primal = ||Ax^{k+1/2} - y^{k+1/2}||
                res_dual = ||A'yt^{k+1/2} + xt^{k+1/2}||

            in C and Python, check that these quantities agree

            (2) calculate solver convergence,

                    res_primal <= tol_primal
                    res_dual <= tol_dual,

                in C and Python, check that the results agree
        """
        f, f_py = test_params.f, test_params.f_py
        g, g_py = test_params.g, test_params.g_py
        residuals = test_params.res
        tolerances = test_params.tol
        objectives = test_params.obj
        local_vars = test_params.z

        f_list = [lib.function(*f_) for f_ in f_py]
        g_list = [lib.function(*g_) for g_ in g_py]

        converged = np.zeros(1, dtype=int)
        cvg_ptr = converged.ctypes.data_as(c_int)

        self.assertCall( lib.pogs_check_convergence(
                solver, objectives, residuals, tolerances, cvg_ptr) );

        self.load_all_local(lib, local_vars, solver)
        obj_py = proxutils.func_eval_python(g_list, local_vars.x12)
        obj_py += proxutils.func_eval_python(f_list, local_vars.y12)
        obj_gap_py = abs(local_vars.z12.dot(local_vars.zt12))
        obj_dua_py = obj_py - obj_gap_py

        tol_primal = tolerances.atolm + (
                tolerances.reltol * np.linalg.norm(local_vars.y12))
        tol_dual = tolerances.atoln + (
                tolerances.reltol * np.linalg.norm(local_vars.xt12))

        self.assertScalarEqual( objectives.gap, obj_gap_py, self.RTOL )
        self.assertScalarEqual( tolerances.primal, tol_primal, self.RTOL )
        self.assertScalarEqual( tolerances.dual, tol_dual, self.RTOL )

        Ax = A_equil.dot(local_vars.x12)
        ATnu = A_equil.T.dot(local_vars.yt12)
        res_primal = np.linalg.norm(Ax - local_vars.y12)
        res_dual = np.linalg.norm(ATnu + local_vars.xt12)

        self.assertScalarEqual( residuals.primal, res_primal, self.RTOL )
        self.assertScalarEqual( residuals.dual, res_dual, self.RTOL )
        self.assertScalarEqual( residuals.gap, abs(obj_gap_py), self.RTOL )

        converged_py = res_primal <= tolerances.primal and \
                       res_dual <= tolerances.dual

        self.assertEqual( converged, converged_py )

    def assert_pogs_unscaling(self, lib, output, solver, local_vars):
        """pogs unscaling test

            solver variables are unscaled and copied to output.

            the following equalities should hold elementwise:

                x^{k+1/2} * e - x_out == 0
                y^{k+1/2} / d - y_out == 0
                -rho * xt^{k+1/2} / e - mu_out == 0
                -rho * yt^{k+1/2} * d - nu_out == 0
        """
        if not isinstance(solver, lib.pogs_solver_p):
            raise TypeError('argument "solver" must be of type {}'.format(
                            lib.pogs_solver_p))

        if not isinstance(output, self.PogsOutputLocal):
            raise TypeError('argument "output" must be of type {}'.format(
                            self.PogsOutputLocal))

        if not isinstance(local_vars, self.PogsVariablesLocal):
            raise TypeError('argument "local_vars" must be of type {}'.format(
                            self.PogsVariablesLocal))

        rho = solver.contents.rho
        suppress = solver.contents.settings.contents.suppress
        self.load_all_local(lib, local_vars, solver)

        if 'pogs_matrix_p' in lib.__dict__:
            solver_work = solver.contents.M
        elif 'pogs_work_p' in lib.__dict__:
            solver_work = solver.contents.W
        else:
            raise ValueError('argument "lib" must contain a field named'
                             '"pogs_matrix_p" or "pogs_work_p"')

        self.assertCall( lib.pogs_unscale_output(
                output.ptr, solver.contents.z, solver_work.contents.d,
                solver_work.contents.e, rho, suppress) )

        self.assertVecEqual(
                local_vars.x12 * local_vars.e, output.x,
                self.ATOLN, self.RTOL )
        self.assertVecEqual(
                local_vars.y12, local_vars.d * output.y,
                self.ATOLM, self.RTOL )
        self.assertVecEqual(
                -rho * local_vars.xt12, local_vars.e * output.mu,
                self.ATOLN, self.RTOL )
        self.assertVecEqual(
                -rho * local_vars.yt12 * local_vars.d, output.nu,
                self.ATOLM, self.RTOL )

    def assert_pogs_convergence(self, lib, A_orig, settings, output):
        RFP = repeat_factor_primal = 2.**(TEST_ITERATE)
        RFD = repeat_factor_dual = 3.**(TEST_ITERATE)

        m, n = self.shape
        rtol = settings.reltol
        atolm = settings.abstol * (m**0.5)
        atoln = settings.abstol * (n**0.5)

        P = 10 * 1.5**int(lib.FLOAT) * 1.5**int(lib.GPU);
        D = 20 * 1.5**int(lib.FLOAT) * 1.5**int(lib.GPU);

        Ax = A_orig.dot(output.x)
        ATnu = A_orig.T.dot(output.nu)
        self.assertVecEqual( Ax, output.y, RFP * atolm, RFP * P * rtol )
        self.assertVecEqual( ATnu, -output.mu, RFD * atoln, RFD * D * rtol )

    def assert_scaled_variables(self, A_equil, x0, nu0, local_vars):
        x_scal = local_vars.e * local_vars.x
        nu_scal = -rho * local_vars.d * local_vars.yt
        y0_unscal = A_equil.dot(x0 / local_vars.e)
        mu0_unscal = A_equil.T.dot(nu0 / (rho * local_vars.d))

        self.assertVecEqual( x0, xscal, self.ATOLN, self.RTOL )
        self.assertVecEqual( nu0, nu_scal, self.ATOLM, self.RTOL )
        self.assertVecEqual( y0_unscal, local_vars.y, self.ATOLM, self.RTOL )
        self.assertVecEqual( mu0_unscal, local_vars.xt, self.ATOLN, self.RTOL )

    def assert_pogs_warmstart_scaling(self, lib, solver, A_equil, test_params):
        m, n = A_equil.shape
        settings = test_params.settings
        local_vars = test_params.z

        local_vars = self.PogsVariablesLocal(m, n, lib.pyfloat)

        # TEST POGS_SET_Z0
        rho = solver.contents.rho
        x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
        nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)
        settings.x0 = x0_ptr
        settings.nu0 = nu0_ptr
        self.assertCall( lib.pogs_update_settings(solver.contents.settings,
                                             ct.byref(settings)) )
        self.assertCall( lib.pogs_set_z0(solver) )
        self.load_all_local(lib, local_vars, solver)
        self.assert_scaled_variables(A_equil, x0, nu0, local_vars)

        # TEST SOLVER WARMSTART
        x0, x0_ptr = self.gen_py_vector(lib, n, random=True)
        nu0, nu0_ptr = self.gen_py_vector(lib, m, random=True)
        settings.x0 = x0_ptr
        settings.nu0 = nu0_ptr
        settings.warmstart = 1
        settings.maxiter = 0

        if self.VERBOSE_TEST:
            print('\nwarm start variable loading test (maxiter = 0)')
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                                        output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )
        self.load_all_local(lib, local_vars, solver)
        self.assert_scaled_variables(x0, nu0, local_vars)

    def assert_warmstart_sequence(self, lib, solver, test_params):
        f = test_params.f
        g = test_params.g
        info = test_params.info
        settings = test_params.settings
        output = test_params.output

        settings.verbose = 1

        # COLD START
        print('\ncold')
        settings.warmstart = 0
        settings.maxiter = MAXITER_DEFAULT
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # REPEAT
        print('\nrepeat')
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # RESUME
        print('\nresume')
        settings.resume = 1
        settings.rho = info.rho
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # WARM START: x0
        print('\nwarm start x0')
        settings.resume = 0
        settings.rho = 1
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, rho
        print('\nwarm start x0, rho')
        settings.resume = 0
        settings.rho = info.rho
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, nu0
        print('\nwarm start x0, nu0')
        settings.resume = 0
        settings.rho = 1
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, nu0
        print('\nwarm start x0, nu0, rho')
        settings.resume = 0
        settings.rho = info.rho
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        self.assertEqual( info.err, 0 )
        self.assertTrue( info.converged or info.k >= settings.maxiter )

    def assert_coldstart_components(self, test_context):
        lib = test_context.lib
        A = test_context.A_dense
        m, n = A.shape
        solver = test_context.solver
        W = solver.contents.W
        params = test_context.params

        self.assertCall( lib.pogs_initialize_conditions(
                    params.obj, params.res, params.tols, params.settings, m, n) )

        self.assert_equilibrate_matrix(lib, W, A)
        A_equil = build_A_equil(lib, W, A)
        self.assert_apply_matrix(lib, W, A_equil)
        self.assert_apply_adjoint(lib, W, A_equil)
        self.assert_project_graph(lib, W, A_equil)
        self.assert_pogs_scaling(lib, solver, params)
        self.assert_pogs_primal_update(lib, solver, params.z)
        self.assert_pogs_prox(lib, solver, params)
        assert_pogs_project_graph(self, lib, solver, A_equil, params.z)
        self.assert_pogs_dual_update(lib, solver, params.z)
        self.assert_pogs_check_convergence(lib, solver, A_equil, params)
        self.assert_pogs_adapt_rho(lib, solver, params)
        self.assert_pogs_unscaling(lib, output, solver, params.z)

    def assert_pogs_call(self, test_context):
        ctx = test_context
        lib = ctx.lib
        solver = ctx.solver
        params = ctx.params
        f, g = params.f, params.g
        settings, info, output = params.settings, params.info, params.output

        self.assertCall( lib.pogs_solve(solver, params.f, params.g, settings, info, output) )
        if info.converged:
            self.assert_pogs_convergence(ctx.A_dense, settings, output)

    def assert_pogs_unified(self, test_context):
        ctx = test_context
        lib = ctx.lib
        data = ctx._data
        flags = ctx.flags
        params = ctx.params
        f, g = params.f, params.g
        settings, info, output = params.settings, params.info, params.output

        settings.verbose = 1
        self.assertCall( lib.pogs(data, flags, f, g, settings, info, output, 0) )
        if info.converged:
            self.assert_pogs_convergence(ctx.A_dense, settings, output)

    def assert_pogs_warmstart(self, test_context):
        lib = test_context.lib
        A = test_context.A_dense
        solver = test_context.solver
        params = test_context.params

        A_equil = build_A_equil(lib, solver.contents.W, A)
        self.assert_pogs_warmstart_scaling(lib, solver, A_equil, params)
        self.assert_warmstart_sequence(lib, solver, params)

    def assert_pogs_io(self, test_context):
        lib = test_context.lib
        A = test_context.A_dense
        solver = test_context.solver
        params = test_context.params
        f, g = params.f, params.g
        settings, info, output = params.settings, params.info, params.output

        m, n = A.shape
        x_rand, _ = self.gen_py_vector(lib, n)
        nu_rand, _ = self.gen_py_vector(lib, m)
        settings.verbose = 1

        # solve
        print('initial solve -> export data')
        self.assertCall( lib.pogs_solve(
                solver, f, g, settings, info, output.ptr) )
        k_orig = info.k

        # placeholders for problem state
        # A_equil, A_equil_ptr = self.gen_py_matrix(lib, m, n, order)
        # if lib.direct:
        #   k = min(m, n)
        #   LLT, LLT_ptr = self.gen_py_matrix(lib, k, k, order)
        # else:
        #   LLT_ptr = LLT = ct.c_void_p()

        # d, d_ptr = self.gen_py_vector(lib, m)
        # e, e_ptr = self.gen_py_vector(lib, n)
        # z, z_ptr = self.gen_py_vector(lib, m + n)
        # z12, z12_ptr = self.gen_py_vector(lib, m + n)
        # zt, zt_ptr = self.gen_py_vector(lib, m + n)
        # zt12, zt12_ptr = self.gen_py_vector(lib, m + n)
        # zprev, zprev_ptr = self.gen_py_vector(lib, m + n)
        # rho, rho_ptr = self.gen_py_vector(lib, 1)

        # # copy state out
        # # TODO: FIX THIS!!!!!!!!!
        # self.assertCall( lib.pogs_extract_solver(
        #       solver, A_equil_ptr, LLT_ptr, d_ptr, e_ptr, z_ptr,
        #       z12_ptr, zt_ptr, zt12_ptr, zprev_ptr, rho_ptr, order) )
        # self.free_var('solver')

        # # copy state in to new solver
        # solver = lib.pogs_load_solver(
        #       A_equil_ptr, LLT_ptr, d_ptr, e_ptr, z_ptr, z12_ptr,
        #       zt_ptr, zt12_ptr, zprev_ptr, rho[0], m, n, order)
        # self.register_solver('solver', solver, lib.pogs_finish)

        # settings.resume = 1
        # print('import data -> solve again')
        # self.assertCall( lib.pogs_solve(solver, f, g, settings, info,
        #                               output.ptr) )
        # self.assertTrue(info.k <= k_orig or not info.converged)