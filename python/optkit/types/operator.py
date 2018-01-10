from optkit.compat import *

import numpy as np
import scipy.sparse as sp
import ctypes as ct

from optkit.libs import error

class OperatorTypes(object):
    def __init__(self, backend, lib):
        accepted_operators = [
                str(np.ndarray), str(sp.csr_matrix), str(sp.csc_matrix),
                str(sp.coo_matrix)]

        class AbstractLinearOperator(object):
            def __del__(self):
                self.release_operator()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                self.release_operator()

            def __init__(self, py_operator):
                if isinstance(py_operator, sp.coo_matrix):
                    print('sparse matrix: converting COO input to CSR')
                    py_operator = py_operator.tocsr()

                self.__py = py_operator
                self.__c_data = None
                self.__c_ptr = None
                self.__free_data = lambda op : None
                self.shape = None

                input_is_sparse = isinstance(
                        py_operator, (sp.csr_matrix, sp.csc_matrix))

                if isinstance(py_operator, np.ndarray):
                    if len(py_operator.shape) != 2:
                        raise ValueError('argument "py_operator" must be a 2-D'
                                         '{} when provided as a {}'.format(
                                         np.ndarray, np.ndarray))

                    m, n = self.shape = self.__py.shape
                    order = lib.enums.get_layout(py_operator)
                    input_ = py_operator.astype(lib.pyfloat)
                    input_ptr = input_.ctypes.data_as(lib.ok_float_p)
                    self.__c_data = lib.matrix(0, 0, 0, None, 0)
                    def build_matrix():
                        return lib.matrix_calloc(self.__c_data, m, n, order)
                    def copy_matrix():
                        return lib.matrix_memcpy_ma(self.__c_data, input_ptr, order)
                    def build_operator():
                        return lib.dense_operator_alloc(self.__c_data)
                    self.__free_data = lib.matrix_free
                elif input_is_sparse:
                    m, n = self.shape = self.__py.shape
                    nnz = self.__py.nnz
                    order = lib.enums.get_layout(py_operator)
                    self.__c_data = lib.sparse_matrix(
                                0, 0, 0, 0, None, None, None, 0)
                    def build_matrix():
                        return lib.sp_matrix_calloc(self.__c_data, m, n, nnz, order)
                    def copy_matrix():
                        hdl = ct.c_void_p()
                        assert error.NO_ERR(lib.sp_make_handle(ct.byref(hdl)))
                        A_ptr_p = py_operator.indptr.ctypes.data_as(lib.ok_int_p)
                        A_ind_p = py_operator.indices.ctypes.data_as(lib.ok_int_p)
                        A_val = py_operator.data.astype(lib.pyfloat)
                        A_val_p = A_val.ctypes.data_as(lib.ok_float_p)
                        assert error.NO_ERR(lib.sp_matrix_memcpy_ma(
                                hdl, self.__c_data, A_val_p, A_ind_p, A_ptr_p))
                        assert error.NO_ERR(lib.sp_destroy_handle(hdl))
                        return 0
                    def build_operator():
                        return lib.sparse_operator_alloc(self.__c_data)
                    self.__free_data = lib.sp_matrix_free
                else:
                    raise TypeError('argument "py_operator" must be one of '
                                    '{}'.format(accepted_operators))

                try:
                    assert error.NO_ERR(build_matrix())
                    backend.increment_cobject_count()
                    assert error.NO_ERR(copy_matrix())
                    self.__c_ptr = build_operator()
                    assert isinstance(self.__c_ptr, lib.abstract_operator_p)
                    backend.increment_cobject_count()
                except:
                    raise

            @property
            def c_ptr(self):
                return self.__c_ptr

            def release_operator(self):
                if isinstance(self.c_ptr, lib.abstract_operator_p):
                    try:
                        assert error.NO_ERR(self.c_ptr.contents.free(
                                self.c_ptr.contents.data))
                    except:
                        raise
                    finally:
                        backend.decrement_cobject_count()
                        self.__c_ptr = None

                if self.__c_data is not None:
                    try:
                        assert error.NO_ERR(self.__free_data(self.__c_data))
                    except:
                        raise
                    finally:
                        self.__c_data = None
                        backend.decrement_cobject_count()

        self.AbstractLinearOperator = AbstractLinearOperator