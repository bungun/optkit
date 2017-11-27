from optkit.compat import *

import numpy as np
import scipy.sparse as sp
import ctypes as ct
import collections

import optkit.libs.enums as enums
from optkit.tests import defs
from optkit.tests.C import statements

NO_ERR = statements.noerr

def lib_contexts(lib_factory):
    return list(filter(
            lambda ctx: ctx.lib is not None,
            map(lambda c:
                    CLibContext(None, lib_factory, c),
                    defs.LIB_CONDITIONS)))

def gen_py_vector(lib, size, random=False):
    v_py = np.zeros(size).astype(lib.pyfloat)
    v_ptr = v_py.ctypes.data_as(lib.ok_float_p)
    if random:
        v_py += np.random.random(size)
    return v_py, v_ptr

def gen_py_matrix(lib, size1, size2, order, random=False):
    pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'
    A_py = np.zeros((size1, size2), order=pyorder).astype(lib.pyfloat)
    A_ptr = A_py.ctypes.data_as(lib.ok_float_p)
    if random:
        A_py += np.random.rand(size1, size2)

        # attempt some matrix conditioning: normalize columns and
        # divide by sqrt(# columns)
        factor = size2**0.5
        for j in xrange(size2):
            A_py[:, j] /= factor * A_py[:, j].dot(A_py[:, j])

    return A_py, A_ptr

class CLibContext:
    def __init__(self, lib, libs=None, conditions=None):
        if libs is not None and conditions is not None:
            lib = libs.get(*conditions)
        self.lib = lib
    def __enter__(self):
        return self.lib
    def __exit__(self, *exc):
        assert NO_ERR( self.lib.ok_device_reset() )

class CVariableContext:
    def __init__(self, alloc, free):
        self._alloc = alloc
        self._free = free
    def __enter__(self):
        return self._alloc()
    def __exit__(self, *exc):
        assert NO_ERR(self._free())

class CArrayContext:
    def __init__(self, c, py, pyptr, build, free):
        self.cptr = self.c = c
        self.py = py
        self.pyptr = pyptr
        self._build = build
        self._free = free
    def __enter__(self):
        self._build()
        return self
    def __exit__(self, *exc):
        assert NO_ERR(self._free())

class CArrayIO:
    def __init__(self, py_array, copy_py2c, copy_c2py):
        self._py2c = copy_py2c
        self._c2py = copy_c2py
        self._pyarray = py_array

    def sync_to_py(self):
        assert NO_ERR( self._c2py(self._pyarray) )

    def sync_to_c(self):
        assert NO_ERR( self._py2c(self._pyarray) )

    def copy_to_py(self, py_array):
        assert NO_ERR( self._c2py(py_array) )

    def copy_to_c(self, py_array):
        assert NO_ERR( self._py2c(py_array) )



class CVectorContext(CArrayContext, CArrayIO):
    def __init__(self, lib, size, random=False):
        v = lib.vector(0, 0, None)
        v_py, v_ptr = gen_py_vector(lib, size, random=random)

        def arr2ptr(arr): return np.ravel(arr).ctypes.data_as(type(v_ptr))
        def py2c(py_array):
            return lib.vector_memcpy_va(v, arr2ptr(py_array), 1)
        def c2py(py_array):
            return lib.vector_memcpy_av(arr2ptr(py_array), v, 1)
        def build():
            assert NO_ERR( lib.vector_calloc(v, size) )
            if random:
                assert NO_ERR(py2c(v_py))
        def free(): return lib.vector_free(v)

        CArrayContext.__init__(self, v, v_py, v_ptr, build, free)
        CArrayIO.__init__(self, v_py, py2c, c2py)


class CIndvectorContext(CArrayContext, CArrayIO):
    def __init__(self, lib, size, random_maxidx=0):
        v = lib.indvector(0, 0, None)
        v_py = np.zeros(size).astype(ct.c_size_t)
        v_ptr = v_py.ctypes.data_as(lib.c_size_t_p)
        random = False
        if random_maxidx > 0:
            random = True
            v_py += (int(random_maxidx) * np.random.random(size)).astype(v_py.dtype)

        def arr2ptr(arr): return np.ravel(arr).ctypes.data_as(type(v_ptr))
        def py2c(py_array):
            return lib.indvector_memcpy_va(v, arr2ptr(py_array), 1)
        def c2py(py_array):
            return lib.indvector_memcpy_av(arr2ptr(py_array), v, 1)
        def build():
            assert NO_ERR( lib.indvector_calloc(v, size) )
            if random:
                assert NO_ERR( py2c(v_py) )
        def free(): return lib.indvector_free(v)

        CArrayContext.__init__(self, v, v_py, v_ptr, build, free)
        CArrayIO.__init__(self, v_py, py2c, c2py)

class CDenseMatrixContext(CArrayContext, CArrayIO):
    def __init__(self, lib, size1, size2, order, random=False):
        A = lib.matrix(0, 0, 0, None, order)
        A_py, A_ptr = gen_py_matrix(lib, size1, size2, order, random=random)

        row = enums.OKEnums.CblasRowMajor
        col = enums.OKEnums.CblasColMajor
        def arr2order(arr): return row if arr.flags.c_contiguous else col
        def arr2ptr(arr): return np.ravel(arr).ctypes.data_as(type(v_ptr))
        def py2c(arr): return lib.matrix_memcpy_ma(A, arr2ptr(arr), arr2order(arr))
        def c2py(arr): return lib.matrix_memcpy_am(arr2ptr(arr), A, arr2order(arr))
        def build():
            assert NO_ERR( lib.matrix_calloc(A, size1, size2, order) )
            if random:
                assert NO_ERR( py2c(A_py) )
        def free(): return lib.matrix_free(A)

        CArrayContext.__init__(self, A, A_py, A_ptr, build, free)
        CArrayIO.__init__(self, A, A_py, py2c, c2py)


class CSparseMatrixContext(CArrayContext):
    def __init__(self, lib, Adense, order):
        A = lib.sparse_matrix(0, 0, 0, 0, None, None, None, order)

        A_py = np.zeros(Adense.shape).astype(lib.pyfloat)
        A_py += Adense
        A_sp = sp.csr_matrix(A_py) if order == lib.enums.CblasRowMajor else \
               sp.csc_matrix(A_py)
        m, n = A_sp.shape
        nnz = A_sp.nnz

        def build(): return lib.sp_matrix_calloc(A, m, n, nnz, order)
        def free(): return lib.sp_matrix_free(A)
        def arr2ptrs(sparse_arr):
            return(
                sparse_arr.data.ctypes.data_as(lib.ok_float_p),
                sparse_arr.indices.ctypes.data_as(lib.ok_int_p),
                sparse_arr.indptr.ctypes.data_as(lib.ok_int_p)
            )

        A_ptr = arr2ptrs(A_sp)

        def py2c(hdl, sp_arr):
            v, i, p = arr2ptrs(sp_arr)
            return lib.sp_matrix_memcpy_ma(hdl, A, v, i, p)
        def c2py(hdl, sp_arr):
            v, i, p = arr2ptrs(sp_arr)
            return lib.sp_matrix_memcpy_ma(hdl, v, i, p, A)

        CArrayContext.__init__(self, A, A_py, A_ptr, build, free)
        self.copy_to_c = py2c
        self.copy_to_py = c2py
        self.sync_to_c = lambda hdl: py2c(hdl, self.py)
        self.sync_to_py = lambda hdl: c2py(hdl, self.py)

class CFunctionVectorContext(CArrayContext, CArrayIO):
    def __init__(self, lib, size):
        self._lib = lib
        f = lib.function_vector(0, None)
        f_py = np.zeros(size, dtype=lib.function)
        f_ptr = f_py.ctypes.data_as(lib.function_p)

        def build(): return lib.function_vector_calloc(f, size)
        def free(): return lib.function_vector_free(f)
        CArrayContext.__init__(self, f, f_py, f_ptr, build, free)

        def arr2ptr(arr): return np.ravel(arr).ctypes.data_as(type(f_ptr))
        def py2c(py_array):
            return lib.function_vector_memcpy_va(f, arr2ptr(py_array))
        def c2py(py_array):
            return lib.function_vector_memcpy_av(arr2ptr(py_array), f)
        CArrayIO.__init__(self, f_py, py2c, c2py)


    @property
    def list(self):
        return [self._lib.function(*f) for f in self.py]

    @property
    def vectors(self):
        f_list = self.list
        FVL = collections.namedtuple('FunctionVectorLists', 'h a b c d e s')
        vecs = dict()
        for attr in ('h', 'a', 'b', 'c', 'd', 'e', 's'):
            vecs[attr] =  np.array([getattr(f, attr) for f in f_list])
        return FVL(**vecs)

class CLinalgContext:
    def __init__(self, make, destroy):
        self._hdl = ct.c_void_p()
        self._destroy = destroy
    def __enter__(self):
        assert NO_ERR(make(ct.byref(self._hdl)))
        return self._hdl
    def __exit__(sef, *exc):
        assert NO_ERR(self._destroy(self._hdl))

class CDenseLinalgContext(CLinalgContext):
    def __init__(self, lib):
        CLinalgContext.__init__(
            self, lib.blas_make_handle, lib.blas_destroy_handle)

class CSparseLinalgContext(CLinalgContext):
    def __init__(self, lib):
        CLinalgContext.__init__(
            self, lib.sp_make_handle, lib.sp_destroy_handle)


def c_operator_context(lib, opkey, A, rowmajor=True):
    if opkey == 'dense':
        return CDenseOperatorContext(lib, A, rowmajor)
    elif opkey == 'sparse':
        return CSparseOperatorContext(lib, A, rowmajor)
    else:
        raise ValueError('invalid operator type')

class CDenseOperatorContext:
    def __init__(self, lib, A_py, rowmajor=True):
        self._lib = lib
        self._A = A_py
        self._order = lib.enums.CblasRowMajor if rowmajor else \
                      lib.enums.CblasColMajor

    def __enter__(self):
        m, n = A.shape
        self.A = CDenseMatrixContext(self._lib, m, n, self._order)
        self.A.__enter__()
        self.A.copy_to_c(self._A)
        self.o = self._lib.dense_operator_alloc(self.A.cptr)
        return self.o

    def __exit__(self, *exc):
        assert NO_ERR(self.o.contents.free(self.o.contents.data))
        self.A.__exit__(*exc)

class CSparseOperatorContext:
    def __init__(self, lib, A_py, rowmajor=True):
        self._lib = lib
        self._A = A_py
        self._order = lib.enums.CblasRowMajor if rowmajor else \
                      lib.enums.CblasColMajor

    def __enter__(self):
        m, n = A.shape
        with CSparseLinalgContext(self._lib) as hdl:
            self.A = CSparseMatrixContext(self._lib, self._A, self._order)
            self.A.__enter__()
            self.A.sync_to_c(self.hdl._hdl)
            self.o = self._lib.sparse_operator_alloc(self.A.cptr)
            return self.o

    def __exit__(self, *exc):
        assert NO_ERR(self.o.contents.free(self.o.contents.data))
        self.A.__exit__(*exc)