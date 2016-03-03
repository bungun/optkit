import unittest
import os
import numpy as np
from ctypes import c_void_p, byref
from optkit.libs import DenseLinsysLibs, EquilibrationLibs, ProjectorLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH

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
			dlibs.append(self.dense_libs.get(single_precision=single_precision,
											 gpu=gpu))
			plibs.append(self.equil_libs.get(
						 		dlibs[-1], single_precision=single_precision,
						 		gpu=gpu))
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
	def project(projectorlib, denselib, blas_handle, order, A, x, y,
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

		pytype = denselib.pyfloat

		m, n = A.shape
		skinny = 1 if m >= n else 0
		normalize_flag = 1 if normalize else 0

		# allocate matrices/vectors
		x_in_c = denselib.vector(0, 0, None)
		x_out_c = denselib.vector(0, 0, None)
		y_in_c = denselib.vector(0, 0, None)
		y_out_c = denselib.vector(0, 0, None)
		A_c = denselib.matrix(0, 0, 0, None, order)

		denselib.vector_calloc(x_in_c, n)
		denselib.vector_calloc(x_out_c, n)
		denselib.vector_calloc(y_in_c, m)
		denselib.vector_calloc(y_out_c, m)
		denselib.matrix_calloc(A_c, m, n, order)

		x_in_ptr = x.astype(pytype).ctypes.data_as(denselib.ok_float_p)
		x_out = np.zeros(n).astype(pytype)
		x_out_ptr = x_out.ctypes.data_as(denselib.ok_float_p)

		y_in_ptr = y.astype(pytype).ctypes.data_as(denselib.ok_float_p)
		y_out = np.zeros(m).astype(pytype)
		y_out_ptr = y_out.ctypes.data_as(denselib.ok_float_p)

		A_ptr = A.astype(pytype).ctypes.data_as(denselib.ok_float_p)
		order_in = denselib.enums.CblasRowMajor if A.flags.c_contiguous else \
				   denselib.enums.CblasColMajor

		# copy inputs
		denselib.vector_memcpy_va(x_in_c, x_in_ptr, 1)
		denselib.vector_memcpy_va(y_in_c, y_in_ptr, 1)
		denselib.matrix_memcpy_ma(A_c, A_ptr, order_in)

		# make projector, project
		P = projectorlib.direct_projector(None, None, 0, skinny,
										  normalize_flag)
		projectorlib.direct_projector_alloc(P, A_c)
		projectorlib.direct_projector_initialize(blas_handle, P, 0)
		projectorlib.direct_projector_project(blas_handle, P, x_in_c,
											  y_in_c, x_out_c, y_out_c)

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

		return x_out, y_out

	@staticmethod
	def equil_project(projectorlib, denselib, equilibration_method, proj_test,
					  blas_handle, order, A, x, y, normalize=False):
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

		pytype = denselib.pyfloat

		A = A.astype(pytype)
		m, n = A.shape
		pyorder = 'C' if order == denselib.enums.CblasRowMajor else 'F'

		A_in_ptr = A.ctypes.data_as(denselib.ok_float_p)
		order_in = denselib.enums.CblasRowMajor if A.flags.c_contiguous else \
				   denselib.enums.CblasColMajor

		# Python matrix + vectors for equilibration output
		A_equil_py = np.zeros((m, n), dtype=pytype, order=pyorder)
		d_py = np.zeros(m, dtype=pytype)
		e_py = np.zeros(n, dtype=pytype)

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
		equilibration_method(blas_handle, A_in_ptr, A_equil, d, e, order_in)
		denselib.matrix_memcpy_am(A_equil_ptr, A_equil, order)
		denselib.vector_memcpy_av(d_ptr, d, 1)
		denselib.vector_memcpy_av(e_ptr, e, 1)


		x_p, y_p = proj_test(projectorlib, denselib, blas_handle, order,
							 A_equil_py, x, y, normalize=normalize)

		denselib.matrix_free(A_equil)
		denselib.vector_free(d)
		denselib.vector_free(e)

		return x_p * e_py, y_p / d_py

	def test_projection(self):
		"""projection test

			(1a) generate random A, x, y
			(1b) optionally, normalize A:
			(2) project (x, y) onto graph y = Ax
		"""
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			plib = self.proj_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)
			if plib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				for normalize in (False, True):
					x_proj, y_proj = self.project(plib, dlib, hdl, order,
												  self.A_test, self.x_test,
												  self.y_test,
												  normalize=normalize)

					self.assertTrue(np.allclose(self.A_test.dot(x_proj),
												y_proj))

			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)


	def test_equilibrated_projection(self):
		"""equlibrated projection test

			(1) generate random A, x, y
			(2) equilibrate A
			(2b) optionally, normalize equilibrated A:
			(3) project (x, y) onto graph y = Ax
		"""
		hdl = c_void_p()

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			elib = self.equil_libs.get(dlib, single_precision=single_precision,
									   gpu=gpu)
			plib = self.proj_libs.get(dlib, single_precision=single_precision,
									  gpu=gpu)
			if plib is None:
				continue

			self.assertEqual(dlib.blas_make_handle(byref(hdl)), 0)

			for rowmajor in (True, False):
				order = dlib.enums.CblasRowMajor if rowmajor else \
						dlib.enums.CblasColMajor

				for normalize in (False, True):
					x_proj, y_proj = self.equil_project(plib, dlib,
												  elib.sinkhorn_knopp,
												  self.project, hdl, order,
												  self.A_test, self.x_test,
												  self.y_test,
												  normalize=normalize)

					self.assertTrue(np.allclose(self.A_test.dot(x_proj),
												y_proj))

			self.assertEqual(dlib.blas_destroy_handle(hdl), 0)
			self.assertEqual(dlib.ok_device_reset(), 0)