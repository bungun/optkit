from optkit.compat import *

import os
import numpy as np
import ctypes as ct
import itertools
import unittest

from optkit.libs.prox import ProxLibs
from optkit.libs import enums
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.tests import defs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx

SCALING = 5
FUNCTIONS = enums.OKFunctionEnums().dict.items()

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal
RELAX_SCAL_EQ = statements.relaxed_scalar_equal
STANDARD_VEC_TOLS = statements.standard_vector_tolerances

def function_vector_values_are(lib, func_ctx, h, a, b, c, d, e, s, tol):
    fh, fa, fb, fc, fd, fe, fs = func_ctx.vectors
    assert VEC_EQ( h, fh, 0, 0 )
    assert VEC_EQ( a, fa, tol, tol )
    assert VEC_EQ( b, fb, tol, tol )
    assert VEC_EQ( c, fc, tol, tol )
    assert VEC_EQ( d, fd, tol, tol )
    assert VEC_EQ( e, fe, tol, tol )
    assert VEC_EQ( s, fs, tol, tol )
    return True

def function_vector_set_constants(lib, func_ctx, h, a, b, c, d, e, s):
    for i in xrange(func_ctx.py.size):
        for idx, val in enumerate((h, a, b, c, d, e, s)):
            func_ctx.py[i][idx] = val
    func_ctx.sync_to_c()

def function_vector_set_vectors(lib, func_ctx, h_value, a, b, c, d, e, s):
    for i in xrange(func_ctx.py.size):
        func_ctx.py[i][0]
        for field_idx, vec in enumerate((a, b, c, d, e, s)):
            func_ctx.py[i][field_idx + 1] = vec[i]
    func_ctx.sync_to_c()

class ProxLibsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = ProxLibs()

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_libs_exist(self):
        libs = []
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            libs.append(self.libs.get(single_precision=single_precision,
                                      gpu=gpu))
        assert any(libs)

    def test_lib_types(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            assert 'function' in dir(lib)
            assert 'function_p' in dir(lib)
            assert 'function_vector' in dir(lib)
            assert 'function_vector_p' in dir(lib)

    def test_version(self):
        for (gpu, single_precision) in defs.LIB_CONDITIONS:
            lib = self.libs.get(single_precision=single_precision, gpu=gpu)
            if lib is None:
                continue

            major = ct.c_int()
            minor = ct.c_int()
            change = ct.c_int()
            status = ct.c_int()

            lib.optkit_version(
                    ct.byref(major), ct.byref(minor), ct.byref(change),
                    ct.byref(status))

            version = defs.version_string(major.value, minor.value,
                                          change.value, status.value)

            assert not version.startswith('0.0.0')
            if defs.VERBOSE_TEST:
                print("proxlib version", version)

class ProxTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
        os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
        self.libs = okcctx.lib_contexts(ProxLibs())

    @classmethod
    def tearDownClass(self):
        os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

    def test_alloc(self):
        m, n = defs.shape()
        for lib in self.libs:
            with lib as lib:
                f = lib.function_vector(0, None)
                assert ( f.size == 0 )

                def f_alloc(): return lib.function_vector_calloc(f, m)
                def f_free(): return lib.function_vector_free(f)
                with okcctx.CVariableContext(f_alloc, f_free):
                    assert ( f.size == m )
                assert ( f.size == 0 )

    def test_io(self):
        m, _ = defs.shape()
        _, hval = FUNCTIONS[int(len(FUNCTIONS) * np.random.random())]
        a = SCALING * np.random.random()
        b = np.random.random()
        c = np.random.random()
        d = np.random.random()
        e = np.random.random()
        s = np.random.random()

        for lib in self.libs:
            with lib as lib:
                TOL, _ = STANDARD_VEC_TOLS(lib, m)

                # initialize to default values
                h0 = enums.OKFunctionEnums().dict['Zero']
                a0, b0, c0, d0, e0, s0 = (1., 0., 1., 0., 0., 1.)

                f = okcctx.CFunctionVectorContext(lib, m)
                with f:
                    for i in xrange(m):
                        for ii, val in enumerate((hval, a, b, c, d, e, s)):
                            f.py[i][ii] = val
                    assert function_vector_values_are(
                            lib, f, hval, a, b, c, d, e, s, TOL)

                    # memcpy af
                    assert NO_ERR( lib.function_vector_memcpy_av(f.pyptr, f.c) )
                    assert function_vector_values_are(
                            lib, f, h0, a0, b0, c0, d0, e0, s0, TOL)

                    # memcpy fa
                    function_vector_set_constants(
                            lib, f, hval, a, b, c, d, e, s)
                    assert NO_ERR( lib.function_vector_memcpy_va(f.c, f.pyptr) )
                    assert NO_ERR( lib.function_vector_memcpy_av(f.pyptr, f.c) )
                    assert function_vector_values_are(
                            lib, f, hval, a, b, c, d, e, s, TOL)

    def test_math(self):
        m, _ = defs.shape()
        a = 1 + np.random.random(m)
        b = 1 + np.random.random(m)
        c = 1 + np.random.random(m)
        d = 1 + np.random.random(m)
        e = 1 + np.random.random(m)
        s = 1 + np.random.random(m)
        # (add 1 above to make sure no divide by zero below)

        hval = 0

        for lib in self.libs:
            with lib as lib:
                TOL, _ = STANDARD_VEC_TOLS(lib, m)

                f = okcctx.CFunctionVectorContext(lib, m)
                v = okcctx.CVectorContext(lib, m, random=True)

                with f, v:
                    function_vector_set_vectors(lib, f, hval, a, b, c, d, e, s)

                    # mul
                    assert NO_ERR( lib.function_vector_mul(f.c, v.c) )
                    f.sync_to_py()
                    assert function_vector_values_are(
                            lib, f, hval, a * v.py, b, c, d * v.py, e * v.py,
                            s, TOL)

                    # div
                    assert NO_ERR( lib.function_vector_div(f.c, v.c) )
                    f.sync_to_py()
                    assert function_vector_values_are(
                            lib, f, hval, a, b, c, d, e, s, TOL)


    def test_func_eval(self):
        m, _ = defs.shape()
        # m = 5
        a = 1 + np.random.random(m)
        b = np.random.random(m)
        c = 1 + np.random.random(m)
        d = np.random.random(m)
        e = np.random.random(m)
        s = 1 + 0.1 * np.random.random(m)
        x_rand = 10 * np.random.random(m)

        for lib, (hkey, hval) in itertools.product(self.libs, FUNCTIONS):
            if defs.VERBOSE_TEST:
                print(hkey)

            # avoid domain errors / poorly conditioned summation
            asymmetric = ('Asymm' in hkey or hkey == 'AbsQuad')
            strict_positive = ('Log' in hkey or 'Entr' in hkey)
            x_test = 1. * x_rand
            if asymmetric:
                x_test -= 5.
            elif strict_positive:
                x_test += 1.
            if 'Exp' in hkey:
                x_test = np.log(0.01 + x_rand)

            with lib as lib:
                RTOL, _ = STANDARD_VEC_TOLS(lib, m)
                repeat_factor = min(4, 2**defs.TEST_ITERATE)
                RTOL *= repeat_factor

                TOLMAX = RTOL
                if asymmetric:
                    TOLMAX = 1e-3
                if 'Exp' in hkey:
                    TOLMAX = 1e-2

                f = okcctx.CFunctionVectorContext(lib, m)
                x = okcctx.CVectorContext(lib, m, value=x_test)

                with f, x:
                    function_vector_set_vectors(lib, f, hval, a, b, c, d, e, s)

                    # function evaluation
                    fval_py = func_eval_python(f.list, x.py)

                    fval_c, fval_c_ptr = okcctx.gen_py_vector(lib, 1)

                    assert NO_ERR( lib.function_eval_vector(f.c, x.c, fval_c_ptr) )
                    if defs.VERBOSE_TEST:
                        print("PY:", fval_py, "\tC:", fval_c)
                    assert RELAX_SCAL_EQ( fval_py, fval_c, RTOL, TOLMAX )


    def test_prox_eval(self):
        m, _ = defs.shape()
        # m = 5
        a = 1 + np.random.random(m)
        b = np.random.random(m)
        c = 1 + np.random.random(m)
        d = np.random.random(m)
        e = np.random.random(m)
        s = 1 + 0.1 * np.random.random(m)
        x_rand = 10 * np.random.random(m)

        for lib, (hkey, hval) in itertools.product(self.libs, FUNCTIONS):
            if defs.VERBOSE_TEST:
                print(hkey)

            # avoid domain errors with randomly generated data
            strict_positive = ('Log' in hkey)
            x_test = 1. * x_rand
            if strict_positive:
                x_test += 0.01
            if 'Exp' in hkey:
                x_test = np.log(0.01 * x_rand)
            if 'Entr' in hkey:
                x_test = np.log(1 + 0.01 * x_rand)

            with lib as lib:
                RTOL, ATOL = STANDARD_VEC_TOLS(lib, m)
                repeat_factor = min(4, 2**defs.TEST_ITERATE)
                RTOL *= repeat_factor
                ATOL *= repeat_factor

                f = okcctx.CFunctionVectorContext(lib, m)
                x = okcctx.CVectorContext(lib, m, value=x_test)
                xout = okcctx.CVectorContext(lib, m)

                with f, x, xout:
                    function_vector_set_vectors(lib, f, hval, a, b, c, d, e, s)

                    # proximal operator evaluation, random rho
                    rho = SCALING * np.random.random()
                    prox_py = prox_eval_python(f.list, rho, x.py)
                    assert NO_ERR( lib.prox_eval_vector(f.c, rho, x.c, xout.c) )
                    xout.sync_to_py()
                    if defs.VERBOSE_TEST:
                        print ("|PY|:", np.linalg.norm(prox_py),
                               "\t|C|:", np.linalg.norm(xout.py))
                    assert VEC_EQ( xout.py, prox_py, ATOL, RTOL )
