from optkit.compat import *

import numpy as np
import scipy.sparse as sp
import ctypes as ct

class OperatorTypes(object):
    def __init__(self, backend, lib):
        accepted_operators = [
                str(np.ndarray), str(sp.csr_matrix), str(sp.csc_matrix),
                str(sp.coo_matrix)]

        class AbstractLinearOperator(object):
            def __del__(self):
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
                    order = lib.enums.CblasRowMajor if \
                            self.__py.flags.c_contiguous else \
                            lib.enums.CblasColMajor
                    input_ = self.__py.astype(lib.pyfloat)
                    input_ptr = input_.ctypes.data_as(lib.ok_float_p)

                    self.__c_data = lib.matrix(0, 0, 0, None, 0)
                    lib.matrix_calloc(self.__c_data, m, n, order)
                    backend.increment_cobject_count()

                    lib.matrix_memcpy_ma(self.__c_data, input_ptr, order)

                    self.__c_ptr = lib.dense_operator_alloc(self.__c_data)
                    # ALTERNATE:
                    # self.__c_ptr = lib.pogs_dense_operator_gen(
                    #       input_ptr, m, n, order)
                    backend.increment_cobject_count()

                    self.__free_data = lib.matrix_free

                elif input_is_sparse:
                    m, n = self.shape = self.__py.shape
                    order = lib.enums.CblasRowMajor if \
                            isinstance(py_operator, sp.csr_matrix) else \
                            lib.enums.CblasColMajor

                    hdl = ct.c_void_p()
                    lib.blas_make_handle(ct.byref(hdl))
                    self.__c_data = lib.sparse_matrix(
                            0, 0, 0, 0, None, None, None, 0)
                    A_ptr_p = self.__py.indptr.ctypes.data_as(lib.ok_int_p)
                    A_ind_p = self.__py.indices.ctypes.data_as(lib.ok_int_p)
                    A_val_p = self.__py.data.ctypes.data_as(lib.ok_float_p)

                    lib.sp_matrix_calloc(
                            self.__c_data, m, n, self.__py.nnz, order)
                    backend.increment_cobject_count()

                    lib.sp_matrix_memcpy_ma(
                            hdl, self.__c_data, A_val_p, A_ind_p, A_ptr_p)

                    lib.blas_destroy_handle(hdl)

                    self.__c_ptr = lib.sparse_operator_alloc(
                            self.__c_data)
                    # ALTERNATE:
                    # self.__c_ptr = lib.pogs_sparse_operator_gen(
                    #       A_val_p, A_ind_p, A_ptr_p, m, n, self.__py.nnz,
                    #       order)
                    backend.increment_cobject_count()

                    self.__free_data = lib.sp_matrix_free

                else:
                    raise TypeError('argument "py_operator" must be one of '
                                    '{}'.format(accepted_operators))


            @property
            def c_ptr(self):
                return self.__c_ptr

            def release_operator(self):
                if isinstance(self.c_ptr, lib.abstract_operator_p):
                    self.c_ptr.contents.free(self.c_ptr.contents.data)
                    backend.decrement_cobject_count()
                    self.__c_ptr = None

                if self.__c_data is not None:
                    self.__free_data(self.__c_data)
                    self.__c_data = None
                    backend.decrement_cobject_count()

        self.AbstractLinearOperator = AbstractLinearOperator