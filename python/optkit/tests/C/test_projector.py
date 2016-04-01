import unittest
import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs, EquilibrationLibs, \
						ProjectorLibs, OperatorLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import optkit.tests.C.operator_helper as op_helper

class ProjectorLibsTestCase(unittest.TestCase):
	"""
	TODO: docstring
	"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.equil_libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def test_libs_exist(self):
		dlibs = []
		plibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(
					single_precision=single_precision, gpu=gpu))
			plibs.append(self.equil_libs.get(
					dlibs[-1], single_precision=single_precision, gpu=gpu))
		self.assertTrue(any(dlibs))
		self.assertTrue(any(plibs))

class DirectProjectorTestCase(unittest.TestCase):
	"""
	TODO: docstring
	"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.equil_libs = EquilibrationLibs()
		self.proj_libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)

		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])


	@staticmethod
	def project(projectorlib, denselib, order, A, x, y,
				normalize=False):
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
		denselib.blas_make_handle(byref(hdl))

		m, n = A.shape
		skinny = 1 if m >= n else 0
		normalize_flag = 1 if normalize else 0

		x = x.astype(denselib.pyfloat)
		x_in_ptr = x.ctypes.data_as(denselib.ok_float_p)
		x_out = np.zeros(n).astype(denselib.pyfloat)
		x_out_ptr = x_out.ctypes.data_as(denselib.ok_float_p)

		y = y.astype(denselib.pyfloat)
		y_in_ptr = y.ctypes.data_as(denselib.ok_float_p)
		y_out = np.zeros(m).astype(denselib.pyfloat)
		y_out_ptr = y_out.ctypes.data_as(denselib.ok_float_p)

		A = A.astype(denselib.pyfloat)
		A_ptr = A.ctypes.data_as(denselib.ok_float_p)
		order_in = denselib.enums.CblasRowMajor if A.flags.c_contiguous else \
				   denselib.enums.CblasColMajor


		# build C inputs
		x_in_c = denselib.vector(0, 0, None)
		denselib.vector_calloc(x_in_c, n)
		denselib.vector_memcpy_va(x_in_c, x_in_ptr, 1)

		y_in_c = denselib.vector(0, 0, None)
		denselib.vector_calloc(y_in_c, m)
		denselib.vector_memcpy_va(y_in_c, y_in_ptr, 1)

		A_c = denselib.matrix(0, 0, 0, None, order)
		denselib.matrix_calloc(A_c, m, n, order)
		denselib.matrix_memcpy_ma(A_c, A_ptr, order_in)

		x_out_c = denselib.vector(0, 0, None)
		denselib.vector_calloc(x_out_c, n)

		y_out_c = denselib.vector(0, 0, None)
		denselib.vector_calloc(y_out_c, m)

		# make projector, project
		P = projectorlib.direct_projector(None, None, 0, skinny,
										  normalize_flag)

		projectorlib.direct_projector_alloc(P, A_c)
		projectorlib.direct_projector_initialize(hdl, P, 0)
		projectorlib.direct_projector_project(hdl, P, x_in_c, y_in_c, x_out_c,
											  y_out_c)


		# copy results
		denselib.vector_memcpy_av(x_out_ptr, x_out_c, 1)
		denselib.vector_memcpy_av(y_out_ptr, y_out_c, 1)

		# free memory
		projectorlib.direct_projector_free(P)
		denselib.matrix_free(A_c)
		denselib.vector_free(x_in_c)
		denselib.vector_free(x_out_c)
		denselib.vector_free(y_in_c)
		denselib.vector_free(y_out_c)

		denselib.blas_destroy_handle(hdl)
		denselib.ok_device_reset()

		return x_out, y_out

	@staticmethod
	def equil_project(projectorlib, denselib, equilibration_method, proj_test,
					  order, A, x, y, normalize=False):
		"""Equilibrated direct projection

			Given matrix A \in R^{m x n}, and input vectors y \in R^m
			and x \in R^n:

			(1) equilibrate A, such that

					A_{equil} = D * A * E,

			(2) project (x, y) onto the graph y = A_{equil} * x

			The equality

				D^{-1} * y_out == A * E * x_out

			should hold elementwise to float/double precision
		"""
		hdl = c_void_p()
		denselib.blas_make_handle(byref(hdl))

		A = A.astype(denselib.pyfloat)
		m, n = A.shape
		pyorder = 'C' if order == denselib.enums.CblasRowMajor else 'F'

		A_in_ptr = A.ctypes.data_as(denselib.ok_float_p)
		order_in = denselib.enums.CblasRowMajor if A.flags.c_contiguous else \
				   denselib.enums.CblasColMajor

		# Python matrix + vectors for equilibration output
		A_equil_py = np.zeros((m, n), dtype=denselib.pyfloat, order=pyorder)
		d_py = np.zeros(m, dtype=denselib.pyfloat)
		e_py = np.zeros(n, dtype=denselib.pyfloat)

		A_equil_ptr = A_equil_py.ctypes.data_as(denselib.ok_float_p)
		d_ptr = d_py.ctypes.data_as(denselib.ok_float_p)
		e_ptr = e_py.ctypes.data_as(denselib.ok_float_p)

		# C matrix + vectors for equilibration output
		A_equil = denselib.matrix(0, 0, 0, None, order)
		d = denselib.vector(0, 0, None)
		e = denselib.vector(0, 0, None)

		denselib.matrix_calloc(A_equil, m, n, order)
		denselib.vector_calloc(d, m)
		denselib.vector_calloc(e, n)

		# equilibrate in C, copy equilibration output back to Python
		equilibration_method(hdl, A_in_ptr, A_equil, d, e, order_in)
		denselib.matrix_memcpy_am(A_equil_ptr, A_equil, order)
		denselib.vector_memcpy_av(d_ptr, d, 1)
		denselib.vector_memcpy_av(e_ptr, e, 1)

		denselib.matrix_free(A_equil)
		denselib.vector_free(d)
		denselib.vector_free(e)

		denselib.blas_destroy_handle(hdl)
		denselib.ok_device_reset()

		x_p, y_p = proj_test(projectorlib, denselib, order, A_equil_py, x, y,
							 normalize=normalize)

		return x_p * e_py, y_p / d_py

	def test_projection(self):
		"""projection test

			(1a) generate random A, x, y
			(1b) optionally, normalize A:
			(2) project (x, y) onto graph y = Ax
		"""
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			DIGITS = 3 if plib.FLOAT else 5


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				for normalize in (False, True):
					x_proj, y_proj = self.project(plib, dlib, order,
												  self.A_test, self.x_test,
												  self.y_test,
												  normalize=normalize)

					self.assertTrue(np.allclose(self.A_test.dot(x_proj),
												y_proj, DIGITS))



	def test_equilibrated_projection(self):
		"""equlibrated projection test

			(1) generate random A, x, y
			(2) equilibrate A
			(2b) optionally, normalize equilibrated A:
			(3) project (x, y) onto graph y = Ax
		"""
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			elib = self.equil_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			DIGITS = 4 if plib.FLOAT else 6


			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				for normalize in (False, True):
					x_proj, y_proj = self.equil_project(plib, dlib,
												  elib.sinkhorn_knopp,
												  self.project, order,
												  self.A_test, self.x_test,
												  self.y_test,
												  normalize=normalize)

					self.assertTrue(np.allclose(self.A_test.dot(x_proj),
												y_proj, DIGITS))

class IndirectProjectorTestCase(unittest.TestCase):
	"""
	TODO: docstring
	"""
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()
		self.op_libs = OperatorLibs()
		self.equil_libs = EquilibrationLibs()
		self.proj_libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.A_test_sparse = self.A_test
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)
			self.A_test_sparse = np.zeros(self.shape)
			self.A_test_sparse += self.A_test
			for i in xrange(self.shape[0]):
				for j in xrange(self.shape[1]):
					if np.random.rand() > 0.4:
						self.A_test_sparse[i, j] *= 0

		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])
		self.nnz = sum(sum(self.A_test_sparse > 0))

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def get_opmethods(self, opkey, denselib, sparselib, operatorlib):
		if opkey == 'dense':
			A = self.A_test
			gen = op_helper.gen_dense_operator
			arg_gen = [denselib, operatorlib, A]
			release = op_helper.release_dense_operator
			arg_release = [denselib, operatorlib]
		elif opkey == 'sparse':
			A = self.A_test_sparse
			gen = op_helper.gen_sparse_operator
			arg_gen = [denselib, sparselib, operatorlib, A]
			release = op_helper.release_sparse_operator
			arg_release = [sparselib, operatorlib]
		else:
			raise ValueError('invalid operator type')

		return (A, gen, arg_gen, release, arg_release)

	def test_alloc_free(self):
		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.op_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			# -----------------------------------------
			# test projector alloc for each operator type defined in
			# self.op_keys
			for op_ in self.op_keys:
				print "test indirect projector alloc, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				p = plib.indirect_projector(None, None)

				plib.indirect_projector_alloc(p, o)
				self.assertNotEqual(p.A, 0)
				self.assertNotEqual(p.cgls_work, 0)
				plib.indirect_projector_free(p)

				release_args += [A, o]
				release_operator(*release_args)

	def test_projection(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.op_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			DIGITS = 7 - 2 * plib.FLOAT - 1 * plib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y in python & C

			# inputs
			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			y = dlib.vector(0, 0, None)
			dlib.vector_calloc(y, m)
			y_ = np.zeros(m).astype(dlib.pyfloat)
			y_ptr = y_.ctypes.data_as(dlib.ok_float_p)

			x_ += self.x_test
			dlib.vector_memcpy_va(x, x_ptr, 1)

			y_ += self.y_test
			dlib.vector_memcpy_va(y, y_ptr, 1)

			# outputs
			x_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(x_out, n)
			x_proj = np.zeros(n).astype(dlib.pyfloat)
			x_p_ptr = x_proj.ctypes.data_as(dlib.ok_float_p)

			y_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(y_out, m)
			y_proj = np.zeros(m).astype(dlib.pyfloat)
			y_p_ptr = y_proj.ctypes.data_as(dlib.ok_float_p)

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "indirect projection, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				p = plib.indirect_projector(None, None)

				plib.indirect_projector_alloc(p, o)
				plib.indirect_projector_project(hdl, p, x, y, x_out, y_out)
				plib.indirect_projector_free(p)

				dlib.vector_memcpy_av(x_p_ptr, x_out, 1)
				dlib.vector_memcpy_av(y_p_ptr, y_out, 1)

				self.assertTrue(
						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
						ATOLM + RTOL * np.linalg.norm(y_proj))

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, y
			dlib.vector_free(x)
			dlib.vector_free(y)
			dlib.blas_destroy_handle(byref(hdl))

class DenseDirectProjectorTestCase(unittest.TestCase):
	"""
	TODO: docstring
	"""
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.proj_libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)

		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])

	def test_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_ = np.zeros(self.shape, order=pyorder).astype(dlib.pyfloat)
				A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
				A = dlib.matrix(0, 0, 0, None, order)

				dlib.matrix_calloc(A, m, n, order)

				A_ += self.A_test
				dlib.matrix_memcpy_ma(A, A_ptr, order)

				p = plib.dense_direct_projector_alloc(A)
				self.assertEqual(p.contents.kind, plib.enums.DENSE_DIRECT)
				self.assertEqual(p.contents.size1, m)
				self.assertEqual(p.contents.size2, n)
				self.assertNotEqual(p.contents.data, 0)
				self.assertNotEqual(p.contents.initialize, 0)
				self.assertNotEqual(p.contents.project, 0)
				self.assertNotEqual(p.contents.free, 0)
				plib.projector_free(p)

				dlib.matrix_free(A)
			dlib.ok_device_reset()


	def test_projection(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			TOL_PLACEHOLDER = 1e-8
			DIGITS = 7 - 2 * plib.FLOAT - 1 * plib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y in python & C

			# inputs
			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			y = dlib.vector(0, 0, None)
			dlib.vector_calloc(y, m)
			y_ = np.zeros(m).astype(dlib.pyfloat)
			y_ptr = y_.ctypes.data_as(dlib.ok_float_p)

			x_ += self.x_test
			dlib.vector_memcpy_va(x, x_ptr, 1)

			y_ += self.y_test
			dlib.vector_memcpy_va(y, y_ptr, 1)

			# outputs
			x_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(x_out, n)
			x_proj = np.zeros(n).astype(dlib.pyfloat)
			x_p_ptr = x_proj.ctypes.data_as(dlib.ok_float_p)

			y_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(y_out, m)
			y_proj = np.zeros(m).astype(dlib.pyfloat)
			y_p_ptr = y_proj.ctypes.data_as(dlib.ok_float_p)

			# -----------------------------------------
			# test projection for each matrix layout
			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor
				pyorder = 'C' if rowmajor else 'F'

				A_ = np.zeros(self.shape, order=pyorder).astype(dlib.pyfloat)
				A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
				A = dlib.matrix(0, 0, 0, None, order)
				dlib.matrix_calloc(A, m, n, order)

				A_ += self.A_test
				dlib.matrix_memcpy_ma(A, A_ptr, order)

				p = plib.dense_direct_projector_alloc(A)
				p.contents.initialize(p.contents.data, 0)

				p.contents.project(p.contents.data, x, y, x_out, y_out,
								   TOL_PLACEHOLDER)
				plib.projector_free(p)

				dlib.vector_memcpy_av(x_p_ptr, x_out, 1)
				dlib.vector_memcpy_av(y_p_ptr, y_out, 1)

				self.assertTrue(
						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
						ATOLM + RTOL * np.linalg.norm(y_proj))

				dlib.matrix_free(A)

			# -----------------------------------------
			# free x, y
			dlib.vector_free(x)
			dlib.vector_free(y)
			dlib.blas_destroy_handle(byref(hdl))
			dlib.ok_device_reset()

class GenericIndirectProjectorTestCase(unittest.TestCase):
	"""
	TODO: docstring
	"""
	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()
		self.op_libs = OperatorLibs()
		self.equil_libs = EquilibrationLibs()
		self.proj_libs = ProjectorLibs()

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.shape = None
		if DEFAULT_MATRIX_PATH is not None:
			try:
				self.A_test = np.load(DEFAULT_MATRIX_PATH)
				self.A_test_sparse = self.A_test
				self.shape = A.shape
			except:
				pass
		if self.shape is None:
			self.shape = DEFAULT_SHAPE
			self.A_test = np.random.rand(*self.shape)
			self.A_test_sparse = np.zeros(self.shape)
			self.A_test_sparse += self.A_test
			for i in xrange(self.shape[0]):
				for j in xrange(self.shape[1]):
					if np.random.rand() > 0.4:
						self.A_test_sparse[i, j] *= 0

		self.x_test = np.random.rand(self.shape[1])
		self.y_test = np.random.rand(self.shape[0])
		self.nnz = sum(sum(self.A_test_sparse > 0))

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def get_opmethods(self, opkey, denselib, sparselib, operatorlib):
		if opkey == 'dense':
			A = self.A_test
			gen = op_helper.gen_dense_operator
			arg_gen = [denselib, operatorlib, A]
			release = op_helper.release_dense_operator
			arg_release = [denselib, operatorlib]
		elif opkey == 'sparse':
			A = self.A_test_sparse
			gen = op_helper.gen_sparse_operator
			arg_gen = [denselib, sparselib, operatorlib, A]
			release = op_helper.release_sparse_operator
			arg_release = [sparselib, operatorlib]
		else:
			raise ValueError('invalid operator type')

		return (A, gen, arg_gen, release, arg_release)

	def test_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.op_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test indirect projector alloc, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				p = plib.indirect_projector_generic_alloc(o)
				self.assertEqual(p.contents.kind, plib.enums.INDIRECT)
				self.assertEqual(p.contents.size1, m)
				self.assertEqual(p.contents.size2, n)
				self.assertNotEqual(p.contents.data, 0)
				self.assertNotEqual(p.contents.initialize, 0)
				self.assertNotEqual(p.contents.project, 0)
				self.assertNotEqual(p.contents.free, 0)
				plib.projector_free(p)

				release_args += [A, o]
				release_operator(*release_args)
			dlib.ok_device_reset()

	def test_projection(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(
					single_precision=single_precision, gpu=gpu)
			slib = self.sparse_libs.get(
					dlib, single_precision=single_precision, gpu=gpu)
			olib = self.op_libs.get(
					dlib, slib, single_precision=single_precision, gpu=gpu)
			plib = self.proj_libs.get(
					dlib, olib, single_precision=single_precision, gpu=gpu)
			if plib is None:
				continue

			TOL_CG = 1e-12
			DIGITS = 7 - 2 * plib.FLOAT - 1 * plib.GPU
			RTOL = 10**(-DIGITS)
			ATOLM = RTOL * m**0.5

			hdl = c_void_p()
			dlib.blas_make_handle(byref(hdl))

			# -----------------------------------------
			# allocate x, y in python & C

			# inputs
			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			y = dlib.vector(0, 0, None)
			dlib.vector_calloc(y, m)
			y_ = np.zeros(m).astype(dlib.pyfloat)
			y_ptr = y_.ctypes.data_as(dlib.ok_float_p)

			x_ += self.x_test
			dlib.vector_memcpy_va(x, x_ptr, 1)

			y_ += self.y_test
			dlib.vector_memcpy_va(y, y_ptr, 1)

			# outputs
			x_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(x_out, n)
			x_proj = np.zeros(n).astype(dlib.pyfloat)
			x_p_ptr = x_proj.ctypes.data_as(dlib.ok_float_p)

			y_out = dlib.vector(0, 0, None)
			dlib.vector_calloc(y_out, m)
			y_proj = np.zeros(m).astype(dlib.pyfloat)
			y_p_ptr = y_proj.ctypes.data_as(dlib.ok_float_p)

			# -----------------------------------------
			# test projection for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "indirect projection, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				p = plib.indirect_projector_generic_alloc(o)
				p.contents.project(p.contents.data, x, y, x_out, y_out, TOL_CG)
				plib.projector_free(p)

				dlib.vector_memcpy_av(x_p_ptr, x_out, 1)
				dlib.vector_memcpy_av(y_p_ptr, y_out, 1)

				self.assertTrue(
						np.linalg.norm(A_.dot(x_proj) - y_proj) <=
						ATOLM + RTOL * np.linalg.norm(y_proj))

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, y
			dlib.vector_free(x)
			dlib.vector_free(y)
			dlib.blas_destroy_handle(byref(hdl))
			dlib.ok_device_reset()