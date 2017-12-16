from optkit.compat import *

import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.operator import include_ok_operator, ok_operator_API

def include_ok_cg(lib, **include_args):
    OptkitLibs.conditional_include(
        lib, 'pcg_helper_p', attach_cg_ctypes, **include_args)

def ok_cg_API(): return ok_operator_API() + [attach_cg_ccalls]

class ConjugateGradientLibs(OptkitLibs):
    def __init__(self):
        OptkitLibs.__init__(self, 'libcg_', ok_cg_API())

def attach_cg_ctypes(lib, single_precision=False):
    include_ok_operator(lib, single_precision=single_precision)

    class cgls_helper(ct.Structure):
        _fields_ = [('p', lib.vector_p),
                    ('q', lib.vector_p),
                    ('r', lib.vector_p),
                    ('s', lib.vector_p),
                    ('norm_s', lib.ok_float),
                    ('norm_s0', lib.ok_float),
                    ('norm_x', lib.ok_float),
                    ('xmax', lib.ok_float),
                    ('alpha', lib.ok_float),
                    ('beta', lib.ok_float),
                    ('delta', lib.ok_float),
                    ('gamma', lib.ok_float),
                    ('gamma_prev', lib.ok_float),
                    ('shrink', lib.ok_float),
                    ('blas_handle', ct.c_void_p)]

    lib.cgls_helper = cgls_helper
    lib.cgls_helper_p = ct.POINTER(lib.cgls_helper)

    class pcg_helper(ct.Structure):
        _fields_ = [('p', lib.vector_p),
                    ('q', lib.vector_p),
                    ('r', lib.vector_p),
                    ('z', lib.vector_p),
                    ('temp', lib.vector_p),
                    ('norm_r', lib.ok_float),
                    ('alpha', lib.ok_float),
                    ('gamma', lib.ok_float),
                    ('gamma_prev', lib.ok_float),
                    ('blas_handle', ct.c_void_p),
                    ('never_solved', ct.c_int)]

    lib.pcg_helper = pcg_helper
    lib.pcg_helper_p = ct.POINTER(lib.pcg_helper)

def attach_cg_ccalls(lib, single_precision=False):
    include_ok_cg(lib, single_precision=single_precision)

    ok_float = lib.ok_float
    vector_p = lib.vector_p
    abstract_operator_p = lib.abstract_operator_p
    cgls_helper_p = lib.cgls_helper_p
    pcg_helper_p = lib.pcg_helper_p

    c_uint_p = ct.POINTER(ct.c_uint)

    # argument types
    lib.cgls_helper_alloc.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.cgls_helper_free.argtypes = [cgls_helper_p]

    lib.cgls_nonallocating.argtypes = [
            cgls_helper_p, abstract_operator_p, vector_p, vector_p, ok_float,
            ok_float, ct.c_size_t, ct.c_int, c_uint_p]
    lib.cgls.argtypes = [
            abstract_operator_p, vector_p, vector_p, ok_float, ok_float,
            ct.c_size_t, ct.c_int, c_uint_p]
    lib.cgls_init.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.cgls_solve.argtypes = [
            ct.c_void_p, abstract_operator_p, vector_p, vector_p, ok_float,
            ok_float, ct.c_size_t, ct.c_int, c_uint_p]
    lib.cgls_finish.argtypes = [ct.c_void_p]
    lib.CGLS_MAXFLAG = 4;

    lib.pcg_helper_alloc.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.pcg_helper_free.argtypes = [pcg_helper_p]

    lib.diagonal_preconditioner.argtypes = [
            abstract_operator_p, vector_p, ok_float]

    lib.pcg_nonallocating.argtypes = [
            pcg_helper_p, abstract_operator_p, abstract_operator_p, vector_p,
            vector_p, ok_float, ok_float, ct.c_size_t, ct.c_int, c_uint_p]
    lib.pcg.argtypes = [
            abstract_operator_p, abstract_operator_p, vector_p, vector_p,
            ok_float, ok_float, ct.c_size_t, ct.c_int, c_uint_p]
    lib.pcg_init.argtypes = [ct.c_size_t, ct.c_size_t]
    lib.pcg_solve.argtypes = [
            ct.c_void_p, abstract_operator_p, abstract_operator_p, vector_p,
            vector_p, ok_float, ok_float, ct.c_size_t, ct.c_int, c_uint_p]
    lib.pcg_finish.argtypes = [ct.c_void_p]

    # return types
    OptkitLibs.attach_default_restype(
            lib.cgls_helper_free,
            lib.cgls_nonallocating,
            lib.cgls,
            lib.cgls_solve,
            lib.cgls_finish,
            lib.diagonal_preconditioner,
            lib.pcg_nonallocating,
            lib.pcg,
            lib.pcg_solve,
            lib.pcg_finish,
    )
    lib.cgls_helper_alloc.restype = cgls_helper_p
    lib.cgls_init.restype = ct.c_void_p
    lib.pcg_helper_alloc.restype = pcg_helper_p
    lib.pcg_helper_free.retype = ct.c_uint
    lib.pcg_init.restype = ct.c_void_p