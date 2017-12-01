from optkit.compat import *

import numpy as np
import ctypes as ct

from optkit.utils import proxutils
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx
import optkit.tests.C.pogs_contexts as pogsctx

ALPHA_DEFAULT = 1.7
RHO_DEFAULT = 1.
MAXITER_DEFAULT = 2000
ABSTOL_DEFAULT = 1e-4
RELTOL_DEFAULT = 1e-3
ADAPTIVE_DEFAULT = 1
GAPSTOP_DEFAULT = 0
WARMSTART_DEFAULT = 0
VERBOSE_DEFAULT = 2
SUPPRESS_DEFAULT = 0
RESUME_DEFAULT = 0

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal
STANDARD_TOLS = statements.standard_tolerances

def settings_are_default(lib):
    settings = lib.pogs_settings()
    TOL = 1e-7
    assert NO_ERR(lib.pogs_set_default_settings(settings))
    assert SCAL_EQ(settings.alpha, ALPHA_DEFAULT, TOL)
    assert SCAL_EQ(settings.rho, RHO_DEFAULT, TOL)
    assert SCAL_EQ(settings.abstol, ABSTOL_DEFAULT, TOL)
    assert SCAL_EQ(settings.reltol, RELTOL_DEFAULT, TOL)
    assert SCAL_EQ(settings.maxiter, MAXITER_DEFAULT, TOL)
    assert SCAL_EQ(settings.verbose, VERBOSE_DEFAULT, TOL)
    assert SCAL_EQ(settings.suppress, SUPPRESS_DEFAULT, TOL)
    assert SCAL_EQ(settings.adaptiverho, ADAPTIVE_DEFAULT, TOL)
    assert SCAL_EQ(settings.gapstop, GAPSTOP_DEFAULT, TOL)
    assert SCAL_EQ(settings.warmstart, WARMSTART_DEFAULT, TOL)
    assert SCAL_EQ(settings.resume, RESUME_DEFAULT, TOL)
    return True

def objectives_are_scaled(lib, solver, test_params):
    f, f_py = test_params.f, test_params.f_py
    g, g_py = test_params.g, test_params.g_py
    local_vars = test_params.z
    m, n = test_params.shape
    RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n)

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
    assert NO_ERR( lib.function_vector_memcpy_va(f, f_py_ptr) )
    assert NO_ERR( lib.function_vector_memcpy_va(g, g_py_ptr) )

    # scale function vector
    assert NO_ERR( lib.pogs_scale_objectives(
            solver.contents.f, solver.contents.g,
            solver.contents.W.contents.d,
            solver.contents.W.contents.e, f, g) )

    # retrieve scaled function vector parameters
    assert NO_ERR( lib.function_vector_memcpy_av(f_py_ptr,
                                                   solver.contents.f) )
    assert NO_ERR( lib.function_vector_memcpy_av(g_py_ptr,
                                                   solver.contents.g) )
    f_list = [lib.function(*f_) for f_ in f_py]
    g_list = [lib.function(*f_) for f_ in g_py]
    f_h1, f_a1, f_b1, f_c1, f_d1, f_e1 = fv_list2arrays(f_list)
    g_h1, g_a1, g_b1, g_c1, g_d1, g_e1 = fv_list2arrays(g_list)

    # retrieve scaling
    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)

    # scaled vars
    assert VEC_EQ( f_a0, local_vars.d * f_a1, ATOLM, RTOL )
    assert VEC_EQ( f_d0, local_vars.d * f_d1, ATOLM, RTOL )
    assert VEC_EQ( f_e0, local_vars.d * f_e1, ATOLM, RTOL )
    assert VEC_EQ( g_a0 * local_vars.e, g_a1, ATOLN, RTOL )
    assert VEC_EQ( g_d0 * local_vars.e, g_d1, ATOLN, RTOL )
    assert VEC_EQ( g_e0 * local_vars.e, g_e1, ATOLN, RTOL )

    # unchanged vars
    assert VEC_EQ( f_h0, f_h1, ATOLM, RTOL )
    assert VEC_EQ( f_b0, f_b1, ATOLM, RTOL )
    assert VEC_EQ( f_c0, f_c1, ATOLM, RTOL )
    assert VEC_EQ( g_h0, g_h1, ATOLN, RTOL )
    assert VEC_EQ( g_b0, g_b1, ATOLN, RTOL )
    assert VEC_EQ( g_c0, g_c1, ATOLN, RTOL )

    return True

def build_A_equil(lib, solver_work, A_orig):
    m, n = A_orig.shape
    d_local = np.zeros(m).astype(lib.pyfloat)
    e_local = np.zeros(n).astype(lib.pyfloat)
    pogsctx.load_to_local(lib, d_local, solver_work.contents.d)
    pogsctx.load_to_local(lib, e_local, solver_work.contents.e)
    return pogsctx.EquilibratedMatrix(d_local, A_orig, e_local,)

def matrix_is_equilibrated(lib, solver_work, A_orig):
    m, n = A_orig.shape
    d_local = np.zeros(m).astype(lib.pyfloat)
    e_local = np.zeros(n).astype(lib.pyfloat)
    pogsctx.load_to_local(lib, d_local, solver_work.contents.d)
    pogsctx.load_to_local(lib, e_local, solver_work.contents.e)
    x = okcctx.CVectorContext(lib, n, random=True)
    y = okcctx.CVectorContext(lib, m)
    RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

    with x, y:
        if lib.py_pogs_impl == 'dense':
            A = solver_work.contents.A
            hdl = solver_work.contents.linalg_handle
            tr = lib.enums.CblasNoTrans
            assert NO_ERR( lib.blas_gemv(hdl, tr, 1, A, x.c, 0, y.c) )
        elif lib.py_pogs_impl == 'abstract':
            opA = solver_work.contents.A
            assert NO_ERR( opA.contents.apply(opA.contents.data, x.c, y.c) )
        else:
            raise ValueError('UNKNOWN POGS IMPLEMENTATION')

        y.sync_to_py()
        DAEX = d_local * A_orig.dot(e_local * x.py)
        return VEC_EQ( y.py, DAEX, ATOLM, RTOL )


def matrix_multiply_works(lib, work, A_equil):
    m, n = A_equil.shape
    x = okcctx.CVectorContext(lib, n, random=True)
    y = okcctx.CVectorContext(lib, m, random=True)
    alpha = np.random.random()
    beta = np.random.random()
    RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

    with x, y:
        apply_A = 'pogs_{}_apply_matrix'.format(lib.py_pogs_impl)
        assert NO_ERR(
                getattr(lib, apply_A)(work, alpha, x.c, beta, y.c))
        AXPY = alpha * A_equil.dot(x.py) + beta * y.py
        y.sync_to_py()
        return VEC_EQ( y.py, AXPY, ATOLM, RTOL )

def adjoint_multiply_works(lib, work, A_equil):
    m, n = A_equil.shape
    x = okcctx.CVectorContext(lib, n, random=True)
    y = okcctx.CVectorContext(lib, m, random=True)
    alpha = np.random.random()
    beta = np.random.random()
    RTOL, _, ATOLN, _ = STANDARD_TOLS(lib, m, n)

    with x, y:
        apply_AT = 'pogs_{}_apply_adjoint'.format(lib.py_pogs_impl)
        assert NO_ERR(
                getattr(lib, apply_AT)(work, alpha, y.c, beta, x.c))
        ATYPX = alpha * A_equil.T.dot(y.py) + beta * x.py
        x.sync_to_py()
        return VEC_EQ( x.py, ATYPX, ATOLN, RTOL )

def graph_projector_works(lib, work, A_equil):
    m, n = A_equil.shape
    RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n)

    x_in = okcctx.CVectorContext(lib, n, random=True)
    y_in = okcctx.CVectorContext(lib, m, random=True)
    x_out = okcctx.CVectorContext(lib, n)
    y_out = okcctx.CVectorContext(lib, m)

    with x_in, y_in, x_out, y_out:
        project = 'pogs_{}_project_graph'.format(lib.py_pogs_impl)
        assert NO_ERR(getattr(
                lib, project)(work, x_in.c, y_in.c, x_out.c, y_out.c, RTOL))

        x_out.sync_to_py()
        y_out.sync_to_py()
        AX = A_equil.dot(x_out.py)
        return VEC_EQ( y_out.py, AX, ATOLM, RTOL)

def primal_update_is_correct(lib, solver, local_vars):
    """primal update test

        set

            z^k = z^{k-1}

        check

            z^k == z^{k-1}

        holds elementwise
    """
    m, n = local_vars.m, local_vars.n
    RTOL, _, _, ATOLMN = STANDARD_TOLS(lib, m, n)
    assert NO_ERR( lib.pogs_primal_update(solver.contents.z) )
    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)
    return VEC_EQ( local_vars.z, local_vars.prev, ATOLMN, RTOL )

def proximal_update_is_correct(lib, solver, test_params):
    """proximal operator application test

        set

            x^{k+1/2} = prox_{g, rho}(x^k - xt^k)
            y^{k+1/2} = prox_{f, rho}(y^k - yt^k)

        in C and Python, check that results agree
    """
    m, n = test_params.shape
    RTOL, ATOLM, ATOLN, _ = STANDARD_TOLS(lib, m, n)

    f, f_py = test_params.f, test_params.f_py
    g, g_py = test_params.g, test_params.g_py
    local_vars = test_params.z
    z = solver.contents.z
    rho = test_params.settings.rho

    hdl = solver.contents.linalg_handle
    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)

    assert NO_ERR( lib.pogs_prox(hdl, f, g, z, rho) )
    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)

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
    assert VEC_EQ( local_vars.x12, x_out, ATOLN, RTOL )
    assert VEC_EQ( local_vars.y12, y_out, ATOLM, RTOL )
    return True

def projection_update_is_correct(lib, solver, A_equil, local_vars):
    """primal projection test
        set
            (x^{k+1}, y^{k+1}) = proj_{y=Ax}(x^{k+1/2}, y^{k+1/2})
        check that
            y^{k+1} == A * x^{k+1}
        holds to numerical tolerance
    """
    m, n = A_equil.shape
    RTOL, ATOLM, _, _ = STANDARD_TOLS(lib, m, n)

    assert NO_ERR( lib.pogs_project_graph(
            solver.contents.W, solver.contents.z,
            solver.contents.settings.contents.alpha, RTOL) )

    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)
    assert VEC_EQ(
            A_equil.dot(local_vars.x), local_vars.y, ATOLM, RTOL)
    return True

def dual_update_is_correct(lib, solver, local_vars):
    """dual update test

        set

            zt^{k+1/2} = z^{k+1/2} - z^{k} + zt^{k}
            zt^{k+1} = zt^{k} - z^{k+1} + alpha * z^{k+1/2} +
                                          (1-alpha) * z^k

        in C and Python, check that results agree
    """
    m, n = local_vars.m, local_vars.n
    RTOL, _, _, ATOLMN = STANDARD_TOLS(lib, m, n)

    blas_handle = solver.contents.linalg_handle
    z = solver.contents.z

    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)
    alpha = solver.contents.settings.contents.alpha
    zt12_py = local_vars.z12 - local_vars.prev + local_vars.zt
    zt_py = local_vars.zt - local_vars.z + (
                alpha * local_vars.z12 + (1-alpha) * local_vars.prev)

    assert NO_ERR( lib.pogs_dual_update(blas_handle, z, alpha) )
    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)
    assert VEC_EQ( local_vars.zt12, zt12_py, ATOLMN, RTOL )
    assert VEC_EQ( local_vars.zt, zt_py, ATOLMN, RTOL )
    return True

def rho_update_is_correct(lib, solver, test_params):
    """adaptive rho test

        rho is rescaled
        dual variable is rescaled accordingly

        the following equality should hold elementwise:

            rho_before * zt_before == rho_after * zt_after

        (here zt, or z tilde, is the dual variable)
    """
    m, n = test_params.shape
    residuals = test_params.res
    tolerances = test_params.tol
    local_vars = test_params.z
    RTOL, ATOLM, ATOLN, ATOLMN = STANDARD_TOLS(lib, m, n)

    rho_p = lib.ok_float_p()
    rho_p.contents = lib.ok_float(solver.contents.rho)

    z = solver.contents.z
    settings = solver.contents.settings

    if settings.contents.adaptiverho:
        rho_params = lib.adapt_params(1.05, 0, 0, 1)
        zt_before = local_vars.zt
        rho_before = solver.contents.rho
        assert NO_ERR( lib.pogs_adapt_rho(
                z, rho_p, rho_params, solver.contents.settings, residuals,
                tolerances, 1) )
        local_vars.load_all_from(solver)
        # pogsctx.load_all_local(lib, local_vars, solver)
        zt_after = local_vars.zt
        rho_after = rho_p.contents
        assert VEC_EQ(
                rho_after * zt_after, rho_before * zt_before, ATOLMN,
                RTOL )
    return True

def convergence_test_is_consistent(lib, solver, A_equil, test_params):
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
    m, n = test_params.shape
    f, f_py = test_params.f, test_params.f_py
    g, g_py = test_params.g, test_params.g_py
    residuals = test_params.res
    tolerances = test_params.tol
    objectives = test_params.obj
    local_vars = test_params.z
    RTOL, ATOLM, ATOLN, ATOLMN = STANDARD_TOLS(lib, m, n)

    f_list = [lib.function(*f_) for f_ in f_py]
    g_list = [lib.function(*g_) for g_ in g_py]

    converged = np.zeros(1, dtype=int)
    cvg_ptr = converged.ctypes.data_as(ct.POINTER(ct.c_int))

    assert NO_ERR( lib.pogs_check_convergence(
            solver, objectives, residuals, tolerances, cvg_ptr) );

    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)
    obj_py = proxutils.func_eval_python(g_list, local_vars.x12)
    obj_py += proxutils.func_eval_python(f_list, local_vars.y12)
    obj_gap_py = abs(local_vars.z12.dot(local_vars.zt12))
    obj_dua_py = obj_py - obj_gap_py

    tol_primal = tolerances.atolm + (
            tolerances.reltol * np.linalg.norm(local_vars.y12))
    tol_dual = tolerances.atoln + (
            tolerances.reltol * np.linalg.norm(local_vars.xt12))

    assert SCAL_EQ( objectives.gap, obj_gap_py, RTOL )
    assert SCAL_EQ( tolerances.primal, tol_primal, RTOL )
    assert SCAL_EQ( tolerances.dual, tol_dual, RTOL )

    Ax = A_equil.dot(local_vars.x12)
    ATnu = A_equil.T.dot(local_vars.yt12)
    res_primal = np.linalg.norm(Ax - local_vars.y12)
    res_dual = np.linalg.norm(ATnu + local_vars.xt12)

    assert SCAL_EQ( residuals.primal, res_primal, RTOL )
    assert SCAL_EQ( residuals.dual, res_dual, RTOL )
    assert SCAL_EQ( residuals.gap, abs(obj_gap_py), RTOL )

    converged_py = res_primal <= tolerances.primal and \
                   res_dual <= tolerances.dual

    return (converged == converged_py)

def output_is_unscaled(lib, output, solver, local_vars):
    """pogs unscaling test

        solver variables are unscaled and copied to output.

        the following equalities should hold elementwise:

            x^{k+1/2} * e - x_out == 0
            y^{k+1/2} / d - y_out == 0
            -rho * xt^{k+1/2} / e - mu_out == 0
            -rho * yt^{k+1/2} * d - nu_out == 0
    """
    m, n = local_vars.m, local_vars.n
    work = solver.contents.W
    rho = solver.contents.rho
    suppress = solver.contents.settings.contents.suppress
    RTOL, ATOLM, ATOLN, ATOLMN = STANDARD_TOLS(lib, m, n)

    local_vars.load_all_from(solver)
    # pogsctx.load_all_local(lib, local_vars, solver)

    assert NO_ERR( lib.pogs_unscale_output(
            output.ptr, solver.contents.z, work.contents.d,
            work.contents.e, rho, suppress) )

    assert VEC_EQ(
            local_vars.x12 * local_vars.e, output.x,
            ATOLN, RTOL )
    assert VEC_EQ(
            local_vars.y12, local_vars.d * output.y,
            ATOLM, RTOL )
    assert VEC_EQ(
            -rho * local_vars.xt12, local_vars.e * output.mu,
            ATOLN, RTOL )
    assert VEC_EQ(
            -rho * local_vars.yt12 * local_vars.d, output.nu,
            ATOLM, RTOL )
    return True

def output_is_converged(lib, A_orig, settings, output):
    RFP = repeat_factor_primal = 2.**(defs.TEST_ITERATE)
    RFD = repeat_factor_dual = 3.**(defs.TEST_ITERATE)

    m, n = A_orig.shape
    rtol = settings.reltol
    atolm = settings.abstol * (m**0.5)
    atoln = settings.abstol * (n**0.5)

    P = 10 * 1.5**int(lib.FLOAT) * 1.5**int(lib.GPU);
    D = 20 * 1.5**int(lib.FLOAT) * 1.5**int(lib.GPU);

    Ax = A_orig.dot(output.x)
    ATnu = A_orig.T.dot(output.nu)
    assert VEC_EQ( Ax, output.y, RFP * atolm, RFP * P * rtol )
    assert VEC_EQ( ATnu, -output.mu, RFD * atoln, RFD * D * rtol )
    return True

def inputs_are_scaled(lib, A_equil, x0, nu0, local_vars, rho):
    m, n = local_vars.m, local_vars.n
    x_scal = local_vars.e * local_vars.x
    nu_scal = -rho * local_vars.d * local_vars.yt
    y0_unscal = A_equil.dot(x0 / local_vars.e)
    mu0_unscal = A_equil.T.dot(nu0 / (rho * local_vars.d))
    RTOL, ATOLM, ATOLN, ATOLMN = STANDARD_TOLS(lib, m, n)

    assert VEC_EQ( x0, x_scal, ATOLN, RTOL )
    assert VEC_EQ( nu0, nu_scal, ATOLM, RTOL )
    assert VEC_EQ( y0_unscal, local_vars.y, ATOLM, RTOL )
    assert VEC_EQ( mu0_unscal, local_vars.xt, ATOLN, RTOL )
    return True

def solver_components_work(test_context):
    lib = test_context.lib
    A = test_context.A_dense
    m, n = A.shape
    params = test_context.params

    with test_context.solver as solver:
        W = solver.contents.W

        assert NO_ERR( lib.pogs_initialize_conditions(
                    params.obj, params.res, params.tol, params.settings, m, n) )

        assert matrix_is_equilibrated(lib, W, A)
        A_equil = build_A_equil(lib, W, A)
        assert matrix_multiply_works(lib, W, A_equil)
        assert adjoint_multiply_works(lib, W, A_equil)
        assert graph_projector_works(lib, W, A_equil)
        assert objectives_are_scaled(lib, solver, params)
        assert primal_update_is_correct(lib, solver, params.z)
        assert proximal_update_is_correct(lib, solver, params)
        assert projection_update_is_correct(lib, solver, A_equil, params.z)
        assert dual_update_is_correct(lib, solver, params.z)
        assert convergence_test_is_consistent(lib, solver, A_equil, params)
        assert rho_update_is_correct(lib, solver, params)
        assert output_is_unscaled(lib, params.output, solver, params.z)
        return True

def residuals_recoverable(test_context):
    lib = test_context.lib
    params = test_context.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output

    settings.maxiter = 100
    settings.diagnostic = 1

    with test_context.solver as solver:
        assert not NO_ERR(lib.pogs_solve(solver, f, g, settings, info, output.ptr))
        # fails w/o memory access errors

    output.attach_convergence_vectors(lib, settings.maxiter)
    with test_context.solver as solver:
        assert NO_ERR(lib.pogs_solve(solver, f, g, settings, info, output.ptr))
        assert np.sum(output.primal_residuals != 0) == info.k
        assert np.sum(output.dual_residuals != 0) == info.k
        assert np.sum(output.primal_tolerances != 0) == info.k
        assert np.sum(output.dual_tolerances != 0) == info.k

    return True

def solve_call_executes(test_context):
    ctx = test_context
    lib = ctx.lib
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output

    with ctx.solver as solver:
        assert NO_ERR(lib.pogs_solve(solver, f, g, settings, info, output.ptr))
        if info.converged:
            assert output_is_converged(lib, ctx.A_dense, settings, output)
        return True

def solver_scales_warmstart_inputs(test_context):
    lib = test_context.lib
    A = test_context.A_dense
    test_params = test_context.params

    with test_context.solver as solver:
        A_equil = build_A_equil(lib, solver.contents.W, A)

        m, n = A_equil.shape
        settings = test_params.settings
        local_vars = test_params.z
        f, g, = test_params.f, test_params.g
        info, output = test_params.info, test_params.output

        # TEST POGS_SET_Z0
        rho = settings.rho
        x0, x0_ptr = okcctx.gen_py_vector(lib, n, random=True)
        nu0, nu0_ptr = okcctx.gen_py_vector(lib, m, random=True)
        settings.x0 = x0_ptr
        settings.nu0 = nu0_ptr
        assert NO_ERR( lib.pogs_update_settings(
                solver.contents.settings, ct.byref(settings)) )
        assert NO_ERR( lib.pogs_set_z0(solver) )
        local_vars.load_all_from(solver)
        # pogsctx.load_all_local(lib, local_vars, solver)
        assert inputs_are_scaled(lib, A_equil, x0, nu0, local_vars, rho)

        # TEST SOLVER WARMSTART
        x0, x0_ptr = okcctx.gen_py_vector(lib, n, random=True)
        nu0, nu0_ptr = okcctx.gen_py_vector(lib, m, random=True)
        settings.x0 = x0_ptr
        settings.nu0 = nu0_ptr
        settings.warmstart = 1
        settings.maxiter = 0

        if defs.VERBOSE_TEST:
            print('\nwarm start variable loading test (maxiter = 0)')
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                                        output.ptr) )
        rho = solver.contents.rho
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )
        local_vars.load_all_from(solver)
        # pogsctx.load_all_local(lib, local_vars, solver)
        return inputs_are_scaled(lib, A_equil, x0, nu0, local_vars, rho)

def warmstart_reduces_iterations(test_context):
    lib = test_context.lib
    solver = test_context.solver
    test_params = test_context.params

    f = test_params.f
    g = test_params.g
    info = test_params.info
    settings = test_params.settings
    output = test_params.output

    settings.verbose = 1

    with test_context.solver as solver:
        # COLD START
        print('\ncold')
        settings.warmstart = 0
        settings.maxiter = MAXITER_DEFAULT
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # REPEAT
        print('\nrepeat')
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # RESUME
        print('\nresume')
        settings.resume = 1
        settings.rho = info.rho
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # WARM START: x0
        print('\nwarm start x0')
        settings.resume = 0
        settings.rho = 1
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, rho
        print('\nwarm start x0, rho')
        settings.resume = 0
        settings.rho = info.rho
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, nu0
        print('\nwarm start x0, nu0')
        settings.resume = 0
        settings.rho = 1
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )

        # WARM START: x0, nu0
        print('\nwarm start x0, nu0, rho')
        settings.resume = 0
        settings.rho = info.rho
        settings.x0 = output.x.ctypes.data_as(lib.ok_float_p)
        settings.nu0 = output.nu.ctypes.data_as(lib.ok_float_p)
        settings.warmstart = 1
        assert NO_ERR( lib.pogs_solve(solver, f, g, settings, info,
                         output.ptr) )
        assert ( info.err == 0 )
        assert ( info.converged or info.k >= settings.maxiter )
    return True

def overrelaxation_reduces_iterations(test_context):
    ctx = test_context
    lib = ctx.lib
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output
    settings.maxiter = 50000
    settings.adaptiverho = 0
    settings.accelerate = 0
    settings.rho = 1.

    k0 = 0

    with ctx.solver as solver:
        settings.alpha = 1.
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        k0 = info.k

    with ctx.solver as solver:
        settings.alpha = 1.7
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        assert info.k < k0
    return True

def adaptive_rho_reduces_iterations(test_context):
    ctx = test_context
    lib = ctx.lib
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output
    settings.maxiter = 50000
    settings.accelerate = 0
    settings.rho = 1.

    k0 = 0

    with ctx.solver as solver:
        settings.adaptiverho = 0
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        k0 = info.k

    with ctx.solver as solver:
        settings.adaptiverho = 1
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        assert info.k < k0
    return True

def anderson_reduces_iterations(test_context):
    ctx = test_context
    lib = ctx.lib
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output
    settings.verbose = 1
    settings.maxiter = 50000
    k0 = 0

    with ctx.solver as solver:
        print('without anderson')
        settings.accelerate = 0
        settings.rho = 1
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        k0 = info.k

    print('with anderson')
    with ctx.solver as solver:
        settings.accelerate = 1
        settings.rho = 1
        assert NO_ERR( lib.pogs_solve(
                solver, params.f, params.g, settings, info, output.ptr) )
        if info.converged:
            assert info.k < k0

    return True

def integrated_pogs_call_executes(test_context):
    ctx = test_context
    lib = ctx.lib
    data = ctx.data
    flags = ctx.flags
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output

    settings.verbose = 1
    assert NO_ERR(lib.pogs(data, flags, f, g, settings, info, output.ptr, 0))
    if info.converged:
        assert output_is_converged(lib, ctx.A_dense, settings, output)
    return True

def solver_data_transferable(test_context):
    ctx = test_context.lib
    lib = ctx.lib
    A = ctx.A_dense
    params = ctx.params
    f, g = params.f, params.g
    settings, info, output = params.settings, params.info, params.output

    m, n = A.shape
    x_rand, _ = okcctx.gen_py_vector(lib, n)
    nu_rand, _ = okcctx.gen_py_vector(lib, m)
    settings.verbose = 1


#     # solve
#     print('initial solve -> export data')
#     assert NO_ERR( lib.pogs_solve(
#             solver, f, g, settings, info, output.ptr) )
#     k_orig = info.k

#     # EXPORT/IMPORT STATE
#     n_state = solver.contents.z.contents.state.contents.size
#     state_out = np.array(n_state, dtype=lib.pyfloat)
#     rho_out = np.array(1, dtype=lib.pyfloat)

#     state_ptr = lib.ok_float_pointerize(state_out)
#     rho_ptr = lib.ok_float_pointerize(rho_out)

#     assert NO_ERR( lib.pogs_solver_save_state(
#             state_ptr, rho_ptr, solver) )

#     build_solver = lambda: lib.pogs_init(ctx.data, ctx.flags)
#     free_solver = lambda s: lib.pogs_finish(s, 0)
#     with okcctx.CVariableContext(build_solver, free_solver) as solver2:
#         settings.resume = 1
#         assert NO_ERR( lib.pogs_solver_load_state(
#                 solver2, state_ptr, rho_ptr[0]))
#         assert NO_ERR( lib.pogs_solve(
#                 solver2, f, g, settings, info, output.ptr) )

#         assert (info.k <= k_orig or not info.converged)

#     # EXPORT/IMPORT SOLVER
#     priv_data = lib.pogs_solver_priv_data()
#     flags = lib.pogs_solver_flags()

#     assert NO_ERR( lib.pogs_export_solver(
#             priv_data, state_ptr, rho_ptr, flags, solver))

#     build_solver = lambda: lib.pogs_load_solver(
#             priv_data, state_ptr, rho_out[0], flags)
#     with okcctx.CVariableContext(build_solver, free_solver) as solver3:
#         settings.resume = 1
#         assert NO_ERR( lib.pogs_solve(
#                 solver3, f, g, settings, info, output.ptr) )
#     return True
