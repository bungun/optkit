from ctypes import CDLL, c_int, c_size_t
from optkit.types import ok_float, ok_float_p
from optkit.types.lowlevel import vector_p, matrix_p
from optkit.defs import SPARSE_TAG as MATRIX__,  GPU_TAG as DEVICE__, \
						FLOAT_TAG as PRECISION__


libpath = '/Users/Baris/Documents/Thesis/modules/optkit/'
oklib = CDLL('{}build/libok_{}_{}{}.dylib'.format(
	libpath,DEVICE__,MATRIX__,PRECISION__))

# TODO: argtypes different for gpu because of cuda handle. i wonder whether this can be pre-pended to the argtypes lists????

# Vector 
# ------
## arguments
oklib.__vector_alloc.argtypes=[vector_p, c_size_t]
oklib.__vector_calloc.argtypes=[vector_p, c_size_t]
oklib.__vector_free.argtypes=[vector_p]
oklib.__vector_set_all.argtypes=[vector_p, ok_float]
oklib.__vector_subvector.argtypes=[vector_p, vector_p, c_size_t, c_size_t]
oklib.__vector_view_array.argtypes=[vector_p, ok_float_p, c_size_t]
oklib.__vector_memcpy_vv.argtypes=[vector_p, vector_p]
oklib.__vector_memcpy_va.argtypes=[vector_p, ok_float_p]
oklib.__vector_memcpy_av.argtypes=[ok_float_p, vector_p]
oklib.__vector_print.argtypes=[vector_p]
oklib.__vector_scale.argtypes=[vector_p, ok_float]
oklib.__vector_add.argtypes=[vector_p, vector_p]
oklib.__vector_sub.argtypes=[vector_p, vector_p]
oklib.__vector_mul.argtypes=[vector_p, vector_p]
oklib.__vector_div.argtypes=[vector_p, vector_p]
oklib.__vector_add_constant.argtypes=[vector_p, ok_float]

## return values
oklib.__vector_alloc.restype=None
oklib.__vector_calloc.restype=None
oklib.__vector_free.restype=None
oklib.__vector_set_all.restype=None
oklib.__vector_subvector.restype=None
oklib.__vector_view_array.restype=None
oklib.__vector_memcpy_vv.restype=None
oklib.__vector_memcpy_va.restype=None
oklib.__vector_memcpy_av.restype=None
oklib.__vector_print.restype=None
oklib.__vector_scale.restype=None
oklib.__vector_add.restype=None
oklib.__vector_sub.restype=None
oklib.__vector_mul.restype=None
oklib.__vector_div.restype=None
oklib.__vector_add_constant.restype=None

# Matrix
# ------
## arguments
oklib.__matrix_alloc.argtypes=[matrix_p, c_size_t, c_size_t, c_int]
oklib.__matrix_calloc.argtypes=[matrix_p, c_size_t, c_size_t, c_int]
oklib.__matrix_free.argtypes=[matrix_p]
oklib.__matrix_submatrix.argtypes=[matrix_p, matrix_p, c_size_t, c_size_t, c_size_t, c_size_t]
oklib.__matrix_row.argtypes=[vector_p, matrix_p, c_size_t]
oklib.__matrix_column.argtypes=[vector_p, matrix_p, c_size_t]
oklib.__matrix_diagonal.argtypes=[vector_p, matrix_p]
oklib.__matrix_view_array.argtypes=[matrix_p, ok_float_p, c_size_t, c_size_t, c_int]
oklib.__matrix_set_all.argtypes=[matrix_p, ok_float]
oklib.__matrix_memcpy_mm.argtypes=[matrix_p, matrix_p]
oklib.__matrix_memcpy_ma.argtypes=[matrix_p, ok_float_p]
oklib.__matrix_memcpy_am.argtypes=[ok_float_p, matrix_p]
oklib.__matrix_print.argtypes=[matrix_p]
oklib.__matrix_scale.argtypes=[matrix_p, ok_float]

## return values 
oklib.__matrix_alloc.restype=None
oklib.__matrix_calloc.restype=None
oklib.__matrix_free.restype=None
oklib.__matrix_submatrix.restype=None
oklib.__matrix_row.restype=None
oklib.__matrix_column.restype=None
oklib.__matrix_diagonal.restype=None
oklib.__matrix_view_array.restype=None
oklib.__matrix_set_all.restype=None
oklib.__matrix_memcpy_mm.restype=None
oklib.__matrix_memcpy_ma.restype=None
oklib.__matrix_memcpy_am.restype=None
oklib.__matrix_print.restype=None
oklib.__matrix_scale.restype=None

# BLAS
# ----
## arguments
oklib.__blas_axpy.argtypes=[ok_float, vector_p, vector_p]
oklib.__blas_nrm2.argtypes=[vector_p]
oklib.__blas_scal.argtypes=[ok_float, vector_p]
oklib.__blas_asum.argtypes=[vector_p]
oklib.__blas_dot.argtypes=[vector_p, vector_p]
oklib.__blas_gemv.argtypes=[c_int, ok_float, matrix_p, vector_p, ok_float, vector_p]
oklib.__blas_trsv.argtypes=[c_int, c_int, c_int, matrix_p, vector_p]
oklib.__blas_syrk.argtypes=[c_int, c_int, ok_float, matrix_p, ok_float, matrix_p]
oklib.__blas_gemm.argtypes=[c_int, c_int, ok_float, matrix_p, matrix_p, ok_float, matrix_p]
oklib.__blas_trsm.argtypes=[c_int, c_int, c_int, c_int, ok_float, matrix_p, matrix_p]

## return values 
oklib.__blas_axpy.restype=None
oklib.__blas_nrm2.restype=ok_float
oklib.__blas_scal.restype=None
oklib.__blas_asum.restype=ok_float
oklib.__blas_dot.restype=ok_float
oklib.__blas_gemv.restype=None
oklib.__blas_trsv.restype=None
oklib.__blas_syrk.restype=None
oklib.__blas_gemm.restype=None
oklib.__blas_trsm.restype=None

# LINALG
# -------
## arguments
oklib.__linalg_cholesky_decomp_noblk.argtypes=[matrix_p]
oklib.__linalg_cholesky_decomp.argtypes=[matrix_p]
oklib.__linalg_cholesky_svx.argtypes=[matrix_p, vector_p]

## return values
oklib.__linalg_cholesky_decomp_noblk.restype=None
oklib.__linalg_cholesky_decomp.restype=None
oklib.__linalg_cholesky_svx.restype=None