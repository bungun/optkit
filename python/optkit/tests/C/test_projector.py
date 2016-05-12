import os
import numpy as np
from ctypes import c_void_p, byref, cast
from optkit.libs.projector import ProjectorLibs
from optkit.tests.defs import OptkitTestCase
from optkit.tests.C.base import OptkitCTestCase, OptkitCOperatorTestCase

class ProjectorLibsTestCase(OptkitTestCase):
	"""
	TODO: docstring
	"""
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue( any(libs) )

class DirectProjectorTestCase(OptkitCTestCase):
	"""
	TODO: docstring
	"""
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProjectorLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])

	def test_projection(self):
		"""projection test

			(1a) generate random A, x, y
			(1b) optionally, normalize A:
			(2) project (x, y) onto graph y = Ax

			Given matrix A \in R^{m x n}, and input vectors y \in R^m
			and x \in R^n:


			(0) optionally, normalize A. divide all entries of A by

				 \sum_{i=1}^m {a_i'a_i} / \sqrt{m},  if m >= n
				 \sum_{j=1}^n {a_j'a_j} / \sqrt{n},  otherwise

			(1) calculate Cholesky factorization of

				(I + AA') if m >= n or
				(I + A'A) otherwise

			(2) set (x_out, y_out) = Proj_{y = Ax} (x, y)


			The equality

				y_out == A * x_out

			should hold elementwise to float/double precision
		"""
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 5 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * self.shape[0]**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				for normalize in (False, True):
					hdl = self.register_blas_handle(lib, 'hdl')

					skinny = 1 if m >= n else 0

					# make Python and C variables
					x_in, xi_, xi_ptr = self.register_vector(lib, n, 'x_in')
					x_out, xo_, xo_ptr = self.register_vector(lib, n, 'x_out')
					y_in, yi_, yi_ptr = self.register_vector(lib, m, 'y_in')
					y_out, yo_, yo_ptr = self.register_vector(lib, m, 'y_out')
					A, A_, A_ptr = self.register_matrix(lib, m, n, order, 'A')

					xi_ += np.random.rand(n)
					yi_ += np.random.rand(m)
					A_ += np.random.rand(m, n)
					order_ = lib.enums.CblasRowMajor if A_.flags.c_contiguous \
							   else lib.enums.CblasColMajor

					# populate C inputs
					self.assertCall( lib.vector_memcpy_va(x_in, xi_ptr, 1) )
					self.assertCall( lib.vector_memcpy_va(y_in, yi_ptr, 1) )
					self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order_) )

					# make projector, project
					P = lib.direct_projector(None, None, 0, skinny, 0)
					self.register_var('P', P, lib.direct_projector_free)

					self.assertCall( lib.direct_projector_alloc(P, A) )
					self.assertCall( lib.direct_projector_initialize(
							hdl, P, normalize) )
					self.assertCall( lib.direct_projector_project(
							hdl, P, x_in, y_in, x_out, y_out) )

					# copy results
					self.assertCall( lib.vector_memcpy_av( xo_ptr, x_out, 1) )
					self.assertCall( lib.vector_memcpy_av( yo_ptr, y_out, 1) )

					# test projection y_out == Ax_out
					if normalize:
						Ax = A_.dot(xo_) / P.normA
					else:
						Ax = A_.dot(xo_)
					self.assertVecEqual( Ax, yo_, ATOLM, RTOL )

					# free memory
					self.free_vars('P', 'A', 'x_in', 'y_in', 'x_out', 'y_out',
								   'hdl')
					self.assertCall( lib.ok_device_reset() )

class IndirectProjectorTestCase(OptkitCOperatorTestCase):

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProjectorLibs()
		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc_free(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			# -----------------------------------------
			# test projector alloc for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				if self.VERBOSE_TEST:
					print "test indirect projector alloc, operator type:", op_
				A_, A, o = self.register_operator(lib, op_)

				p = lib.indirect_projector(None, None)
				self.assertCall( lib.indirect_projector_alloc(p, o) )
				self.register_var('p', p, lib.indirect_projector_free)

				self.assertNotEqual( p.A, 0 )
				self.assertNotEqual( p.cgls_work, 0 )

				self.free_vars('p', 'A', 'o')
				self.assertCall( lib.ok_device_reset() )

	def test_projection(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				if self.VERBOSE_TEST:
					print "indirect projection, operator type:", op_

				hdl = self.register_blas_handle(lib, 'hdl')

				# allocate inputs/outputs in python & C
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				y, y_, y_ptr = self.register_vector(lib, m, 'y')
				x_out, x_proj, x_p_ptr = self.register_vector(lib, n, 'x_out')
				y_out, y_proj, y_p_ptr = self.register_vector(lib, m, 'y_out')

				x_ += self.x_test
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				y_ += self.y_test
				self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )

				A_, A, o = self.register_operator(lib, op_)

				p = lib.indirect_projector(None, None, 0)
				self.assertCall( lib.indirect_projector_alloc(p, o) )
				self.register_var('p', p, lib.indirect_projector_free)
				self.assertCall( lib.indirect_projector_project(
						hdl, p, x, y, x_out, y_out) )
				self.free_var('p')

				self.assertCall( lib.vector_memcpy_av(x_p_ptr, x_out, 1) )
				self.assertCall( lib.vector_memcpy_av(y_p_ptr, y_out, 1) )

				self.assertVecEqual( A_.dot(x_proj), y_proj, ATOLM, RTOL )

				self.free_vars('A', 'o', 'x', 'y', 'x_out', 'y_out', 'hdl')
				self.assertCall( lib.ok_device_reset() )

class DenseDirectProjectorTestCase(OptkitCTestCase):

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProjectorLibs()
		self.A_test = self.A_test_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				A, A_, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				A_ += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				p = lib.dense_direct_projector_alloc(A)
				self.register_var('p', p.contents.data, p.contents.free)
				self.assertEqual( p.contents.kind, lib.enums.DENSE_DIRECT)
				self.assertEqual( p.contents.size1, m)
				self.assertEqual( p.contents.size2, n)
				self.assertNotEqual(p.contents.data, 0)
				self.assertNotEqual(p.contents.initialize, 0)
				self.assertNotEqual(p.contents.project, 0)
				self.assertNotEqual(p.contents.free, 0)

				self.free_vars('p', 'A')
				self.assertCall( lib.ok_device_reset() )

	def test_projection(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			TOL_PLACEHOLDER = 1e-8
			DIGITS = 7 - 3 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			# -----------------------------------------
			# test projection for each matrix layout
			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				hdl = self.register_blas_handle(lib, 'hdl')

				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				y, y_, y_ptr = self.register_vector(lib, m, 'y')
				x_out, x_proj, x_p_ptr = self.register_vector(lib, n, 'x_out')
				y_out, y_proj, y_p_ptr = self.register_vector(lib, m, 'y_out')

				x_ += self.x_test
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				y_ += self.y_test
				self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )

				A, A_, A_ptr = self.register_matrix(lib, m, n, order, 'A')
				A_ += self.A_test
				self.assertCall( lib.matrix_memcpy_ma(A, A_ptr, order) )

				p = lib.dense_direct_projector_alloc(A)
				self.register_var('p', p.contents.data, p.contents.free)
				self.assertCall( p.contents.initialize(p.contents.data, 0) )

				self.assertCall( p.contents.project(
						p.contents.data, x, y, x_out, y_out, TOL_PLACEHOLDER) )

				self.free_var('p')

				self.assertCall( lib.vector_memcpy_av(x_p_ptr, x_out, 1) )
				self.assertCall( lib.vector_memcpy_av(y_p_ptr, y_out, 1) )

				self.assertVecEqual( A_.dot(x_proj), y_proj, ATOLM, RTOL )

				self.free_vars('A', 'x', 'y', 'x_out', 'y_out', 'hdl')
				self.assertCall( lib.ok_device_reset() )

class GenericIndirectProjectorTestCase(OptkitCOperatorTestCase):
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ProjectorLibs()
		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def test_alloc_free(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				if self.VERBOSE_TEST:
					print "test indirect projector alloc, operator type:", op_
				_, A, o = self.register_operator(lib, op_)

				p = lib.indirect_projector_generic_alloc(o)
				self.register_var('p', p.contents.data, p.contents.free)
				self.assertEqual( p.contents.kind, lib.enums.INDIRECT )
				self.assertEqual( p.contents.size1, m )
				self.assertEqual( p.contents.size2, n )
				self.assertNotEqual( p.contents.data, 0 )
				self.assertNotEqual( p.contents.initialize, 0 )
				self.assertNotEqual( p.contents.project, 0 )
				self.assertNotEqual( p.contents.free, 0 )
				self.free_vars('p', 'A', 'o')
				self.assertCall( lib.ok_device_reset() )

	def test_projection(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			TOL_CG = 1e-12
			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				if self.VERBOSE_TEST:
					print "indirect projection, operator type:", op_

				hdl = self.register_blas_handle(lib, 'hdl')

				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				y, y_, y_ptr = self.register_vector(lib, m, 'y')
				x_out, x_proj, x_p_ptr = self.register_vector(lib, n, 'x_out')
				y_out, y_proj, y_p_ptr = self.register_vector(lib, m, 'y_out')

				x_ += self.x_test
				self.assertCall( lib.vector_memcpy_va(x, x_ptr, 1) )

				y_ += self.y_test
				self.assertCall( lib.vector_memcpy_va(y, y_ptr, 1) )

				A_, A, o = self.register_operator(lib, op_)

				p = lib.indirect_projector_generic_alloc(o)
				self.register_var('p', p.contents.data, p.contents.free)
				self.assertCall( p.contents.project(
						p.contents.data, x, y, x_out, y_out, TOL_CG) )
				self.free_var('p')

				self.assertCall( lib.vector_memcpy_av(x_p_ptr, x_out, 1) )
				self.assertCall( lib.vector_memcpy_av(y_p_ptr, y_out, 1) )

				self.assertVecEqual( A_.dot(x_proj), y_proj, ATOLM, RTOL )

				self.free_vars('A', 'o', 'x', 'y', 'x_out', 'y_out', 'hdl')
				self.assertCall( lib.ok_device_reset() )