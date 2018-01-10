from optkit.compat import *

import ctypes as ct

from optkit.libs import loader
from optkit.libs import enums
from optkit.libs.linsys import include_ok_dense, include_ok_sparse, \
        ok_linsys_API

def include_ok_operator(lib, **include_args):
    loader.OptkitLibs.conditional_include(
        lib, 'abstract_operator_p', attach_operator_ctypes, **include_args)

def ok_operator_API(): return ok_linsys_API() + [attach_operator_ccalls]

class OperatorLibs(loader.OptkitLibs):
    def __init__(self):
        loader.OptkitLibs.__init__(self, 'liboperator_', ok_operator_API())

def attach_operator_ctypes(lib, single_precision=False):
    include_ok_dense(lib, single_precision=single_precision)
    include_ok_sparse(lib, single_precision=single_precision)

    ok_float = lib.ok_float
    vector_p = lib.vector_p

    class ok_abstract_operator(ct.Structure):
        _fields_ = [('size1', ct.c_size_t),
                    ('size2', ct.c_size_t),
                    ('data', ct.c_void_p),
                    ('apply', ct.CFUNCTYPE(
                            ct.c_uint, ct.c_void_p, vector_p, vector_p)),
                    ('adjoint', ct.CFUNCTYPE(
                            ct.c_uint, ct.c_void_p,vector_p, vector_p)),
                    ('fused_apply', ct.CFUNCTYPE(
                            ct.c_uint, ct.c_void_p, ok_float, vector_p,
                            ok_float, vector_p)),
                    ('fused_adjoint', ct.CFUNCTYPE(
                            ct.c_uint, ct.c_void_p, ok_float, vector_p,
                            ok_float, vector_p)),
                    ('free', ct.CFUNCTYPE(ct.c_uint, ct.c_void_p)),
                    ('kind', ct.c_uint)]

    lib.abstract_operator = ok_abstract_operator
    lib.abstract_operator_p = ct.POINTER(lib.abstract_operator)
    lib.operator_enums = enums.OkOperatorEnums()

    def get_opcode(abstract_operator):
        if isinstance(abstract_operator, lib.abstract_operator_p):
            return abstract_operator.contents.kind
        elif isinstance(abstract_operator, lib.abstract_operator):
            return abstract_operator.kind
        else:
            raise TypeError('input must be of type {} or {}'.format(
                    lib.abstract_operator, lib.abstract_operator_p))
    lib.get_opcode = get_opcode

def attach_operator_ccalls(lib, single_precision=False):
    include_ok_operator(lib, single_precision=single_precision)

    # argument types
    lib.dense_operator_alloc.argtypes = [lib.matrix_p]
    lib.sparse_operator_alloc.argtypes = [lib.sparse_matrix_p]
    lib.diagonal_operator_alloc.argtypes = [lib.vector_p]

    # return types
    lib.dense_operator_alloc.restype = lib.abstract_operator_p
    lib.sparse_operator_alloc.restype = lib.abstract_operator_p
    lib.diagonal_operator_alloc.restype = lib.abstract_operator_p