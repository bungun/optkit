import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import ProjectorLibs
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
		self.assertTrue(any(libs))

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

	@staticmethod
	def project(lib, order, A, x, y, normalize=False):
		"""Direct projection

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
		hdl = c_void_p()
		lib.blas_make_handle(byref(hdl))

		m, n = A.shape
		skinny = 1 if m >= n else 0
		normalize_flag = 1 if normalize else 0

		x = x.astype(lib.pyfloat)
		x_in_ptr = x.ctypes.data_as(lib.ok_float_p)
		x_out = np.zeros(n).astype(lib.pyfloat)
		x_out_ptr = x_out.ctypes.data_as(lib.ok_float_p)

		y = y.astype(lib.pyfloat)
		y_in_ptr = y.ctypes.data_as(lib.ok_float_p)
		y_out = np.zeros(m).astype(lib.pyfloat)
		y_out_ptr = y_out.ctypes.data_as(lib.ok_float_p)

		A = A.astype(lib.pyfloat)
		A_ptr = A.ctypes.data_as(lib.ok_float_p)
		order_in = lib.enums.CblasRowMajor if A.flags.c_contiguous else \
				   lib.enums.CblasColMajor

		# build C inputs
		x_in_c = lib.vector(0, 0, None)
		lib.vector_calloc(x_in_c, n)
		lib.vector_memcpy_va(x_in_c, x_in_ptr, 1)

		y_in_c = lib.vector(0, 0, None)
		lib.vector_calloc(y_in_c, m)
		lib.vector_memcpy_va(y_in_c, y_in_ptr, 1)

		A_c = lib.matrix(0, 0, 0, None, order)
		lib.matrix_calloc(A_c, m, n, order)
		lib.matrix_memcpy_ma(A_c, A_ptr, order_in)

		x_out_c = lib.vector(0, 0, None)
		lib.vector_calloc(x_out_c, n)

		y_out_c = lib.vector(0, 0, None)
		lib.vector_calloc(y_out_c, m)

		# make projector, project
		P = lib.direct_projector(None, None, 0, skinny, normalize_flag)

		lib.direct_projector_alloc(P, A_c)
		lib.direct_projector_initialize(hdl, P, 0)
		lib.direct_projector_project(hdl, P, x_in_c, y_in_c, x_out_c, y_out_c)

		# copy results
		lib.vector_memcpy_av(x_out_ptr, x_out_c, 1)
		lib.vector_memcpy_av(y_out_ptr, y_out_c, 1)

		# free memory
		lib.direct_projector_free(P)
		lib.matrix_free(A_c)
		lib.vector_free(x_in_c)
		lib.vector_free(x_out_c)
		lib.vector_free(y_in_c)
		lib.vector_free(y_out_c)

		lib.blas_destroy_handle(hdl)
		lib.ok_device_reset()

		return x_out, y_out

	# @staticmethod
	# def equil_project(lib, equilibration_method, proj_test, order, A, x, y,
	# 				  normalize=False):
	# 	"""Equilibrated direct projection

	# 		Given matrix A \in R^{m x n}, and input vectors y \in R^m
	# 		and x \in R^n:

	# 		(1) equilibrate A, such that

	# 				A_{equil} = D * A * E,

	# 		(2) project (x, y) onto the graph y = A_{equil} * x

	# 		The equality

	# 			D^{-1} * y_out == A * E * x_out

	# 		should hold elementwise to float/double precision
	# 	"""
	# 	hdl = c_void_p()
	# 	lib.blas_make_handle(byref(hdl))

	# 	A = A.astype(lib.pyfloat)
	# 	m, n = A.shape
	# 	pyorder = 'C' if order == lib.enums.CblasRowMajor else 'F'

	# 	A_in_ptr = A.ctypes.data_as(lib.ok_float_p)
	# 	order_in = lib.enums.CblasRowMajor if A.flags.c_contiguous else \
	# 			   lib.enums.CblasColMajor

	# 	# Python matrix + vectors for equilibration output
	# 	A_equil_py = np.zeros((m, n), dtype=lib.pyfloat, order=pyorder)
	# 	d_py = np.zeros(m, dtype=lib.pyfloat)
	# 	e_py = np.zeros(n, dtype=lib.pyfloat)

	# 	A_equil_ptr = A_equil_py.ctypes.data_as(lib.ok_float_p)
	# 	d_ptr = d_py.ctypes.data_as(lib.ok_float_p)
	# 	e_ptr = e_py.ctypes.data_as(lib.ok_float_p)

	# 	# C matrix + vectors for equilibration output
	# 	A_equil = lib.matrix(0, 0, 0, None, order)
	# 	d = lib.vector(0, 0, None)
	# 	e = lib.vector(0, 0, None)

	# 	lib.matrix_calloc(A_equil, m, n, order)
	# 	lib.vector_calloc(d, m)
	# 	lib.vector_calloc(e, n)

	# 	# equilibrate in C, copy equilibration output back to Python
	# 	equilibration_method(hdl, A_in_ptr, A_equil, d, e, order_in)
	# 	lib.matrix_memcpy_am(A_equil_ptr, A_equil, order)
	# 	lib.vector_memcpy_av(d_ptr, d, 1)
	# 	lib.vector_memcpy_av(e_ptr, e, 1)

	# 	lib.matrix_free(A_equil)
	# 	lib.vector_free(d)
	# 	lib.vector_free(e)

	# 	lib.blas_destroy_handle(hdl)
	# 	lib.ok_device_reset()

	# 	x_p, y_p = proj_test(lib, lib, order, A_equil_py, x, y,
	# 						 normalize=normalize)

	# 	return x_p * e_py, y_p / d_py

	def test_projection(self):
		"""projection test

			(1a) generate random A, x, y
			(1b) optionally, normalize A:
			(2) project (x, y) onto graph y = Ax
		"""
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 5 - 2 * single_precision
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * self.shape[0]**0.5

			for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
				for normalize in (False, True):
					x_proj, y_proj = self.project(lib, order, self.A_test,
												  self.x_test, self.y_test,
												  normalize=normalize)
					Ax_proj = self.A_test.dot(x_proj)
					print Ax_proj - y_proj
					print y_proj
					print x_proj

					self.assertTrue( np.linalg.norm(Ax_proj - y_proj) <=
									 ATOLM + RTOL * np.linalg.norm(y_proj) )

	# def test_equilibrated_projection(self):
	# 	"""equlibrated projection test

	# 		(1) generate random A, x, y
	# 		(2) equilibrate A
	# 		(2b) optionally, normalize equilibrated A:
	# 		(3) project (x, y) onto graph y = Ax
	# 	"""
	# 	for (gpu, single_precision) in self.CONDITIONS:
	# 		lib = self.libs.get(single_precision=single_precision, gpu=gpu)
	# 		if lib is None:
	# 			continue

	# 		DIGITS = 6 - 2 * single_precision

	# 		for order in (lib.enums.CblasRowMajor, lib.enums.CblasColMajor):
	# 			for normalize in (False, True):
	# 				x_proj, y_proj = self.equil_project(lib,
	# 													lib.sinkhorn_knopp,
	# 											  		self.project, order,
	# 											  		self.A_test, self.x_test,
	# 											 		self.y_test,
	# 											  		normalize=normalize)

	# 				self.assertTrue(np.allclose(self.A_test.dot(x_proj),
	# 											y_proj, DIGITS))

class IndirectProjectorTestCase(OptkitCOperatorTestCase):
	"""
	TODO: docstring
	"""
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

	def test_alloc_free(self):
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			# -----------------------------------------
			# test projector alloc for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "test indirect projector alloc, operator type:", op_
				A_, A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				p = lib.indirect_projector(None, None)

				lib.indirect_projector_alloc(p, o)
				self.register_var('p', p, lib.indirect_projector_free)

				self.assertNotEqual(p.A, 0)
				self.assertNotEqual(p.cgls_work, 0)

				self.free_var('p')
				self.free_var('A')
				self.free_var('o')

	def test_projection(self):
		m, n = self.shape
		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			hdl = c_void_p()
			lib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y in python & C

			# inputs
			x = lib.vector(0, 0, None)
			lib.vector_calloc(x, n)
			self.register_var('x', x, lib.vector_free)
			x_ = np.zeros(n).astype(lib.pyfloat)
			x_ptr = x_.ctypes.data_as(lib.ok_float_p)

			y = lib.vector(0, 0, None)
			lib.vector_calloc(y, m)
			self.register_var('y', y, lib.vector_free)
			y_ = np.zeros(m).astype(lib.pyfloat)
			y_ptr = y_.ctypes.data_as(lib.ok_float_p)

			x_ += self.x_test
			lib.vector_memcpy_va(x, x_ptr, 1)

			y_ += self.y_test
			lib.vector_memcpy_va(y, y_ptr, 1)

			# outputs
			x_out = lib.vector(0, 0, None)
			lib.vector_calloc(x_out, n)
			self.register_var('x_out', x_out, lib.vector_free)
			x_proj = np.zeros(n).astype(lib.pyfloat)
			x_p_ptr = x_proj.ctypes.data_as(lib.ok_float_p)

			y_out = lib.vector(0, 0, None)
			lib.vector_calloc(y_out, m)
			self.register_var('y_out', y_out, lib.vector_free)
			y_proj = np.zeros(m).astype(lib.pyfloat)
			y_p_ptr = y_proj.ctypes.data_as(lib.ok_float_p)

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "indirect projection, operator type:", op_
				A_, A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				p = lib.indirect_projector(None, None)

				lib.indirect_projector_alloc(p, o)
				self.register_var('p', p, lib.indirect_projector_free)
				lib.indirect_projector_project(hdl, p, x, y, x_out, y_out)
				self.free_var('p')

				lib.vector_memcpy_av(x_p_ptr, x_out, 1)
				lib.vector_memcpy_av(y_p_ptr, y_out, 1)

				self.assertTrue(
						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
						ATOLM + RTOL * np.linalg.norm(y_proj))

				self.free_var('A')
				self.free_var('o')

			# -----------------------------------------
			# free x, y
			self.free_var('x')
			self.free_var('y')
			self.free_var('x_out')
			self.free_var('y_out')

			lib.blas_destroy_handle(byref(hdl))

class DenseDirectProjectorTestCase(OptkitCTestCase):
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

	def tearDown(self):
		self.free_all_vars()

	def test_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			for rowmajor in (True, False):
				order = lib.enums.CblasRowMajor if rowmajor else \
						lib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_ = np.zeros(self.shape, order=pyorder).astype(lib.pyfloat)
				A_ptr = A_.ctypes.data_as(lib.ok_float_p)
				A = lib.matrix(0, 0, 0, None, order)

				lib.matrix_calloc(A, m, n, order)
				self.register_var('A', A, lib.matrix_free)

				A_ += self.A_test
				lib.matrix_memcpy_ma(A, A_ptr, order)

				p = lib.dense_direct_projector_alloc(A)
				print p
				print p.contents
				# self.register_var('p', p, p.contents.free)
				self.assertEqual(p.contents.kind, lib.enums.DENSE_DIRECT)
				# self.assertEqual(p.contents.size1, m)
				# self.assertEqual(p.contents.size2, n)
				# self.assertNotEqual(p.contents.data, 0)
				# self.assertNotEqual(p.contents.initialize, 0)
				# self.assertNotEqual(p.contents.project, 0)
				# self.assertNotEqual(p.contents.free, 0)

				# self.free_var('p')
				self.free_var('A')

			lib.ok_device_reset()

# 	def test_projection(self):
# 		m, n = self.shape
# 		for (gpu, single_precision) in self.CONDITIONS:
# 			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
# 			if lib is None:
# 				continue

# 			TOL_PLACEHOLDER = 1e-8
# 			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
# 			RTOL = 10**(-DIGITS)
# 			ATOLM = RTOL * m**0.5

# 			hdl = c_void_p()
# 			lib.blas_make_handle(byref(hdl))

# 			# -----------------------------------------
# 			# allocate x, y in python & C

# 			# inputs
# 			x = lib.vector(0, 0, None)
# 			lib.vector_calloc(x, n)
# 			self.register_var('x', x, lib.vector_free)
# 			x_ = np.zeros(n).astype(lib.pyfloat)
# 			x_ptr = x_.ctypes.data_as(lib.ok_float_p)

# 			y = lib.vector(0, 0, None)
# 			lib.vector_calloc(y, m)
# 			self.register_var('y', y, lib.vector_free)
# 			y_ = np.zeros(m).astype(lib.pyfloat)
# 			y_ptr = y_.ctypes.data_as(lib.ok_float_p)

# 			x_ += self.x_test
# 			lib.vector_memcpy_va(x, x_ptr, 1)

# 			y_ += self.y_test
# 			lib.vector_memcpy_va(y, y_ptr, 1)

# 			# outputs
# 			x_out = lib.vector(0, 0, None)
# 			lib.vector_calloc(x_out, n)
# 			self.register_var('x_out', x_out, lib.vector_free)

# 			x_proj = np.zeros(n).astype(lib.pyfloat)
# 			x_p_ptr = x_proj.ctypes.data_as(lib.ok_float_p)

# 			y_out = lib.vector(0, 0, None)
# 			lib.vector_calloc(y_out, m)
# 			self.register_var('y_out', y_out, lib.vector_free)

# 			y_proj = np.zeros(m).astype(lib.pyfloat)
# 			y_p_ptr = y_proj.ctypes.data_as(lib.ok_float_p)

# 			# -----------------------------------------
# 			# test projection for each matrix layout
# 			for rowmajor in (True, False):
# 				order = lib.enums.CblasRowMajor if rowmajor else \
# 						lib.enums.CblasColMajor
# 				pyorder = 'C' if rowmajor else 'F'

# 				A_ = np.zeros(self.shape, order=pyorder).astype(lib.pyfloat)
# 				A_ptr = A_.ctypes.data_as(lib.ok_float_p)
# 				A = lib.matrix(0, 0, 0, None, order)
# 				lib.matrix_calloc(A, m, n, order)
# 				self.register_var('A', A, lib.matrix_free)

# 				A_ += self.A_test
# 				lib.matrix_memcpy_ma(A, A_ptr, order)

# 				p = lib.dense_direct_projector_alloc(A)
# 				self.register_var('p', p, p.contents.free)
# 				p.contents.initialize(p.contents.data, 0)

# 				p.contents.project(p.contents.data, x, y, x_out, y_out,
# 								   TOL_PLACEHOLDER)

# 				self.free_var('p')

# 				lib.vector_memcpy_av(x_p_ptr, x_out, 1)
# 				lib.vector_memcpy_av(y_p_ptr, y_out, 1)

# 				self.assertTrue(
# 						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
# 						ATOLM + RTOL * np.linalg.norm(y_proj))

# 				self.free_var('A')

# 			# -----------------------------------------
# 			# free x, y
# 			self.free_var('x')
# 			self.free_var('y')
# 			self.free_var('x_out')
# 			self.free_var('y_out')
# 			lib.blas_destroy_handle(byref(hdl))
# 			lib.ok_device_reset()

# class GenericIndirectProjectorTestCase(OptkitCOperatorTestCase):
# 	"""
# 	TODO: docstring
# 	"""
# 	@classmethod
# 	def setUpClass(self):
# 		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
# 		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
# 		self.libs = ProjectorLibs()
# 		self.A_test = self.A_test_gen
# 		self.A_test_sparse = self.A_test_sparse_gen

# 	@classmethod
# 	def tearDownClass(self):
# 		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

# 	def setUp(self):
# 		self.x_test = np.random.rand(self.shape[1])
# 		self.y_test = np.random.rand(self.shape[0])

# 	def tearDown(self):
# 		self.free_all_vars()

# 	def test_alloc_free(self):
# 		m, n = self.shape
# 		for (gpu, single_precision) in self.CONDITIONS:
# 			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
# 			if lib is None:
# 				continue

# 			# -----------------------------------------
# 			# test projection for each operator type defined in self.op_keys
# 			for op_ in self.op_keys:
# 				print "test indirect projector alloc, operator type:", op_
# 				_, A, o, freeA = self.gen_operator(op_, lib)
# 				self.register_var('A', A, freeA)
# 				self.register_var('o', o, o.contents.free)

# 				p = lib.indirect_projector_generic_alloc(o)
# 				self.register_var('p', p, p.contents.free)
# 				self.assertEqual(p.contents.kind, lib.enums.INDIRECT)
# 				self.assertEqual(p.contents.size1, m)
# 				self.assertEqual(p.contents.size2, n)
# 				self.assertNotEqual(p.contents.data, 0)
# 				self.assertNotEqual(p.contents.initialize, 0)
# 				self.assertNotEqual(p.contents.project, 0)
# 				self.assertNotEqual(p.contents.free, 0)
# 				self.free_var('p')

# 				self.free_var('A')
# 				self.free_var('o')

# 			lib.ok_device_reset()

# 	def test_projection(self):
# 		m, n = self.shape
# 		for (gpu, single_precision) in self.CONDITIONS:
# 			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
# 			if lib is None:
# 				continue

# 			TOL_CG = 1e-12
# 			DIGITS = 7 - 2 * lib.FLOAT - 1 * lib.GPU
# 			RTOL = 10**(-DIGITS)
# 			ATOLM = RTOL * m**0.5

# 			hdl = c_void_p()
# 			lib.blas_make_handle(byref(hdl))

# 			# -----------------------------------------
# 			# allocate x, y in python & C

# 			# inputs
# 			x = lib.vector(0, 0, None)
# 			lib.vector_calloc(x, n)
# 			self.register_var('x', x, lib.vector_free)
# 			x_ = np.zeros(n).astype(lib.pyfloat)
# 			x_ptr = x_.ctypes.data_as(lib.ok_float_p)

# 			y = lib.vector(0, 0, None)
# 			lib.vector_calloc(y, m)
# 			self.register_var('y', y, lib.vector_free)
# 			y_ = np.zeros(m).astype(lib.pyfloat)
# 			y_ptr = y_.ctypes.data_as(lib.ok_float_p)

# 			x_ += self.x_test
# 			lib.vector_memcpy_va(x, x_ptr, 1)

# 			y_ += self.y_test
# 			lib.vector_memcpy_va(y, y_ptr, 1)

# 			# outputs
# 			x_out = lib.vector(0, 0, None)
# 			lib.vector_calloc(x_out, n)
# 			self.register_var('x_out', x_out, lib.vector_free)
# 			x_proj = np.zeros(n).astype(lib.pyfloat)
# 			x_p_ptr = x_proj.ctypes.data_as(lib.ok_float_p)

# 			y_out = lib.vector(0, 0, None)
# 			lib.vector_calloc(y_out, m)
# 			self.register_var('y_out', y_out, lib.vector_free)
# 			y_proj = np.zeros(m).astype(lib.pyfloat)
# 			y_p_ptr = y_proj.ctypes.data_as(lib.ok_float_p)

# 			# -----------------------------------------
# 			# test projection for each operator type defined in self.op_keys
# 			for op_ in self.op_keys:
# 				print "indirect projection, operator type:", op_
# 				A_, A, o, freeA = self.gen_operator(op_, lib)
# 				self.register_var('A', A, freeA)
# 				self.register_var('o', o, o.contents.free)

# 				p = lib.indirect_projector_generic_alloc(o)
# 				self.register_var('p', p, p.contents.free)
# 				p.contents.project(p.contents.data, x, y, x_out, y_out, TOL_CG)
# 				self.free_var('p')

# 				lib.vector_memcpy_av(x_p_ptr, x_out, 1)
# 				lib.vector_memcpy_av(y_p_ptr, y_out, 1)

# 				self.assertTrue(
# 						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
# 						ATOLM + RTOL * np.linalg.norm(y_proj))

# 				self.free_var('A')
# 				self.free_var('o')

# 			# -----------------------------------------
# 			# free x, y
# 			self.free_var('x')
# 			self.free_var('y')
# 			self.free_var('x_out')
# 			self.free_var('y_out')
# 			lib.blas_destroy_handle(byref(hdl))
# 			lib.ok_device_reset()