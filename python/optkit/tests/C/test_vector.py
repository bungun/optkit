from optkit.compat import *

import os
import numpy as np
import ctypes as ct

from optkit.libs.linsys import DenseLinsysLibs
from optkit.tests.C import statements
import optkit.tests.C.context_managers as okcctx
from optkit.tests.C.base import OptkitCTestCase

NO_ERR = statements.noerr
VEC_EQ = statements.vec_equal
SCAL_EQ = statements.scalar_equal

class VectorTestCase(OptkitCTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = okcctx.lib_contexts(DenseLinsysLibs())
		# self.libs = DenseLinsysLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc(self):
		for lib in self.libs:
			with lib as lib:
				len_v = 10 + int(10 * np.random.rand())
				v = lib.vector(0, 0, None)
				assert ( v.size == 0 )
				assert ( v.stride == 0 )

				def v_alloc(): return lib.vector_calloc(v, len_v)
				def v_free(): return lib.vector_free(v)
				with okcctx.CVariableContext(v_alloc, v_free):
					assert ( v.size == len_v )
					assert ( v.stride == 1 )
					if not lib.GPU:
						for i in xrange(v.size):
							assert ( v.data[i] == 0 )

				assert ( v.size == 0 )
				assert ( v.stride == 0 )

	def test_io(self):
		for lib in self.libs:
			with lib as lib:
				len_v = 10 + int(1000 * np.random.rand())
				RTOL, ATOL = statements.standard_vector_tolerances(lib, len_v)

				v = okcctx.CVectorContext(lib, len_v)
				w = okcctx.CVectorContext(lib, len_v)
				with v, w:
					# set_all
					set_val = 5
					assert NO_ERR( lib.vector_set_all(v.cptr, set_val) )

					# memcpy_av
					assert NO_ERR( lib.vector_memcpy_av(v.pyptr, v.cptr, 1) )
					assert VEC_EQ( set_val, v.py, ATOL, RTOL )

					# memcpy_va
					w_rand = np.random.random(len_v)
					w.py[:] = w_rand[:]
					assert NO_ERR( lib.vector_memcpy_va(w.cptr, w.pyptr, 1) )
					w.py *= 0
					assert NO_ERR( lib.vector_memcpy_av(w.pyptr, w.cptr, 1) )
					assert VEC_EQ( w.py, w_rand, ATOL, RTOL )

					# memcpy_vv
					assert NO_ERR( lib.vector_memcpy_vv(v.cptr, w.cptr) )
					assert NO_ERR( lib.vector_memcpy_av(v.pyptr, v.cptr, 1) )
					assert VEC_EQ( v.py, w_rand, ATOL, RTOL )

					# view_array
					if not lib.GPU:
						u_rand, u_ptr = okcctx.gen_py_vector(lib, len_v, True)
						u = lib.vector(0, 0, None)
						assert NO_ERR( lib.vector_view_array(u, u_ptr, u_rand.size) )
						assert NO_ERR( lib.vector_memcpy_av(v.pyptr, u, 1) )
						assert VEC_EQ( v.py, u_rand, ATOL, RTOL )
						# DON'T FREE u, DATA OWNED BY PYTHON

	def test_subvector(self):
		for lib in self.libs:
			with lib as lib:
				len_v = 10 + int(10 * np.random.rand())
				v = okcctx.CVectorContext(lib, len_v)

				offset_sub = 3
				len_sub = 3
				v_sub = lib.vector(0, 0, None)

				with v:
					assert NO_ERR( lib.vector_subvector(
							v_sub, v.c, offset_sub, len_sub))
					assert ( v_sub.size == 3 )
					assert ( v_sub.stride == v.c.stride )

					v_sub_py, v_sub_ptr = okcctx.gen_py_vector(lib, len_sub)
					v.sync_to_py()
					assert NO_ERR( lib.vector_memcpy_av(v_sub_ptr, v_sub, 1) )

					v_py_sub = v.py[offset_sub : offset_sub + len_sub]
					assert VEC_EQ(v_sub_py, v_py_sub, 1e-7, 1e-7)

	def test_math(self):
		for lib in self.libs:
			with lib as lib:
				val1 = 12 * np.random.rand()
				val2 = 5 * np.random.rand()
				len_v = 10 + int(1000 * np.random.rand())
				RTOL, ATOL = statements.standard_vector_tolerances(lib, len_v, 1)

				v = okcctx.CVectorContext(lib, len_v)
				w = okcctx.CVectorContext(lib, len_v)

				with v, w:
					def sync_vw_to_py():
						map(lambda v_: v_.sync_to_py(), (v, w))
					def values_equal_expected(val_v, val_w):
						assert VEC_EQ( val1, v.py, ATOL, RTOL )
						assert VEC_EQ( val2, w.py, ATOL, RTOL )
						return True
					def transform_test(vec, transform, val):
						assert NO_ERR(transform(vec.c))
						vec.sync_to_py()
						assert VEC_EQ(val, vec.py, ATOL, RTOL)

					# constant addition
					assert NO_ERR( lib.vector_add_constant(v.c, val1) )
					assert NO_ERR( lib.vector_add_constant(w.c, val2) )
					sync_vw_to_py()
					assert values_equal_expected(val1, val2)

					# add two vectors
					assert NO_ERR( lib.vector_add(v.c, w.c) )
					val1 += val2
					sync_vw_to_py()
					assert values_equal_expected(val1, val2)

					# subtract two vectors
					assert NO_ERR( lib.vector_sub(w.c, v.c) )
					val2 -= val1
					sync_vw_to_py()
					assert values_equal_expected(val1, val2)

					# multiply two vectors
					assert NO_ERR( lib.vector_mul(v.c, w.c) )
					val1 *= val2
					sync_vw_to_py()
					assert values_equal_expected(val1, val2)

					# vector scale
					scal = 3 * np.random.rand()
					val1 *= scal
					transform_test(v, lambda v_: lib.vector_scale(v_, scal), val1)

					# make sure v is strictly positive...
					val1 = 0.5 + abs(np.random.rand())
					assert NO_ERR( lib.vector_scale(v.c, 0) )
					assert NO_ERR( lib.vector_add_constant(v.c, val1) )

					# ...then divide two vectors
					assert NO_ERR( lib.vector_div(w.c, v.c) )
					val2 /= float(val1)
					sync_vw_to_py()
					assert values_equal_expected(val1, val2)

					# vector abs
					w_max = w.py.max()
					def _abs(w_):
						assert NO_ERR( lib.vector_add_constant(w_, -(w_max + 1)) )
						return lib.vector_abs(w_)
					val2 = abs(val2 - (w_max + 1))
					transform_test(w, _abs, val2)

					# vector recip
					val2 = 1. / val2
					transform_test(w, lib.vector_recip, val2)

					# vector sqrt
					val2 **= 0.5
					transform_test(w, lib.vector_sqrt, val2)

					# vector pow
					pow_val = -2 + 4 * np.random.rand()
					val2 **= pow_val
					transform_test(w, lambda w_: lib.vector_pow(w_, pow_val), val2)

					# vector exp
					val2 = np.exp(val2)
					transform_test(w, lib.vector_exp, val2)

					# min / max
					w.py *= 0
					w.py += np.random.random(len(w.py))
					w.sync_to_c()

					# vector argmin
					wargmin = np.zeros(1).astype(ct.c_size_t)
					wargmin_p = wargmin.ctypes.data_as(lib.c_size_t_p)
					assert NO_ERR( lib.vector_indmin(w.c, wargmin_p) )
					assert SCAL_EQ( w.py[wargmin[0]], w.py.min(), RTOL )

					# # vector min
					wmin = np.zeros(1).astype(lib.pyfloat)
					wmin_p = wmin.ctypes.data_as(lib.ok_float_p)
					assert NO_ERR( lib.vector_min(w.c, wmin_p) )
					assert SCAL_EQ( wmin[0], w.py.min(), RTOL )

					# # vector max
					wmax = wmin
					wmax_p = wmin_p
					assert NO_ERR( lib.vector_max(w.c, wmax_p) )
					assert SCAL_EQ( wmax[0], w.py.max(), RTOL )

	def test_indvector_math(self):
		for lib in self.libs:
			with lib as lib:
				len_v = 10 + int(1000 * np.random.rand())
				# len_v = 10 + int(10 * np.random.rand())

				RTOL, _ = statements.standard_vector_tolerances(lib, len_v, 1)

				w = okcctx.CIndvectorContext(lib, len_v, random_maxidx=30)

				with w:
					# vector argmin
					wargmin = np.zeros(1).astype(ct.c_size_t)
					wargmin_p = wargmin.ctypes.data_as(lib.c_size_t_p)
					assert NO_ERR( lib.indvector_indmin(w.c, wargmin_p) )
					assert SCAL_EQ( w.py[wargmin[0]], w.py.min(), RTOL )

					# # vector min
					wmin = wargmin
					wmin_p = wargmin_p
					assert NO_ERR( lib.indvector_min(w.c, wmin_p) )
					assert SCAL_EQ( wmin[0], w.py.min(), RTOL )

					# vector max
					wmax = wmin
					wmax_p = wmin_p
					assert NO_ERR( lib.indvector_max(w.c, wmax_p) )
					assert SCAL_EQ( wmax[0], w.py.max(), RTOL )

