import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs, OperatorLibs, \
						ConjugateGradientLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import pdb

class ConjugateGradientLibsTestCase(unittest.TestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.dense_libs = DenseLinsysLibs()
		self.sparse_libs = SparseLinsysLibs()
		self.op_libs = OperatorLibs()
		self.cg_libs = ConjugateGradientLibs()

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
				if np.random.rand() > 0.4:
					self.A_test_sparse[i, :] *= 0
			for j in xrange(self.shape[1]):
				if np.random.rand() > 0.4:
					self.A_test_sparse[:, j] *= 0

		self.x_test = np.random.rand(self.shape[1])
		self.nnz = sum(sum(self.A_test_sparse > 0))

	@staticmethod
	def calculate_diagonal_preconditioner(A, p, rho):
	    for j in xrange(A.shape[1]):
	    	p[j] = 1. / (rho +  np.linalg.norm(A[:, j])**2)

	@staticmethod
	def gen_preconditioning_operator(dlib, olib, A_py, rho):
		p_vec = dlib.vector(0, 0, None)
		dlib.vector_alloc(p_vec, n)
		p_ = np.zeros(n).astype(dlib.pyfloat)
		p_ptr = p_.ctypes.data_as(dlib.ok_float_p)
		self.calculate_diagonal_preconditioner(A_py, p_, rho)
		dlib.vector_memcpy_va(p_vec, p_ptr, 1)
		p = olib.diagonal_operator_alloc(p_vec)
		return p_vec, p

	@staticmethod
	def release_preconditioning_operator(dlib, olib, p_vec, p):
		olib.operator_free(p)
		dlib.vector_free(p_vec)

 	@staticmethod
 	def gen_dense_operator(dlib, olib, A_py, rowmajor=True):
 		order = dlib.enums.CblasRowMajor if rowmajor else \
 				dlib.enums.CblasColMajor
		pyorder = 'C' if rowmajor else 'F'
		A_ = zeros(A_py.shape, order=pyorder).astype(dlib.pyfloat)
		A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
		A = dlib.matrix(0, 0, 0, None, order)
		dlib.matrix_calloc(A, m, n, order)
		dlib.matrix_memcpy_ma(A, A_ptr, order)
		return A, o

	@staticmethod
	def release_dense_operator(dlib, olib, A, o):
		olib.operator_free(o)
		slib.matrix_free(A)

    @staticmethod
    def gen_sparse_operator(dlib, slib, olib, A_py, rowmajor=True):
 		order = dlib.enums.CblasRowMajor if rowmajor else \
 				dlib.enums.CblasColMajor
		pyorder = 'C' if rowmajor else 'F'
    	sparsemat = csr_matrix if rowmajor else csc_matrix
		A_ = zeros(A_py.shape, order=pyorder).astype(dlib.pyfloat)
		A_ += A_py
		A_sp = csr_matrix(A_)
		A_ptr = A_sp.indptr.ctypes.data_as(slib.ok_int_p)
		A_ind = A_sp.indices.ctypes.data_as(slib.ok_int_p)
		A_val = A_sp.data.ctypes.data_as(dlib.ok_float_p)
		A = slib.matrix(0, 0, 0, 0, None, None, None, order)
		slib.sp_matrix_alloc(A, m, n, A_sp.nnz, order)
		slib.sp_matrix_memcpy_ma(A, A_val, A_ind, A_ptr, order)
		o = olib.sparse_operator_alloc(A)
		return A, o

	@staticmethod
	def release_sparse_operator(slib, olib, A, o)
		olib.operator_free(o)
		slib.matrix_free(A)

	def test_libs_exist(self):
		dlibs = []
		slibs = []
		oplibs = []
		cglibs = []
		for (gpu, single_precision) in CONDITIONS:
			dlibs.append(self.dense_libs.get(gpu=gpu,
						 single_precision=single_precision))
			slibs.append(self.sparse_libs.get(
								dlibs[-1], single_precision=single_precision,
								gpu=gpu))
			oplibs.append(self.op_libs.get(
						  		dlibs[-1], slibs[-1],
						  		single_precision=single_precision, gpu=gpu))
			oplibs.append(self.op_libs.get(
						  		dlibs[-1], slibs[-1],
						  		single_precision=single_precision, gpu=gpu))
			cglibs.append(self.cg_libs.get(
						  		dlibs[-1], oplibs[-1],
						  		single_precision=single_precision, gpu=gpu))

		self.assertTrue(any(dlibs))
		self.assertTrue(any(slibs))
		self.assertTrue(any(oplibs))
		self.assertTrue(any(cglibs))

	def test_cgls_helper_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			h = lib.cgls_helper_alloc(self.shape[0], self.shape[1])
			self.assertTrue(isinstance(h.contents.p, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.q, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.r, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.s, dlib.vector_p))
			lib.cgls_helper_free(h)

	def test_cgls_nonallocating(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			print "TEST CGLS NONALLOCATING"

	def test_cgls_allocating(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			print "TEST CGLS ALLOCATING"

	def test_cgls_easy(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			print "TEST CGLS EASY"

	def test_pcg_helper_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			h = lib.pcg_helper_alloc(self.shape[0], self.shape[1])
			self.assertTrue(isinstance(h.contents.p, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.q, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.r, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.z, dlib.vector_p))
			self.assertTrue(isinstance(h.contents.temp, dlib.vector_p))
			lib.pcg_helper_free(h)

	def test_pcg_nonallocating(self):
		"""

		pcg method attemps to solve

			M(rho * I + A'A)x = b

		to specified tolerance _tol_ within _maxiter_ CG iterations,
		where A is given as an abstract linear operator.
		"""
		tol = 1e-6
		rho = 1e-4
		maxiter = 50

		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5


			# -----------------------------------------
			# allocate x, b in python & C
			b = dlib.vector(0, 0, None)
			dlib.vector_alloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_alloc(b, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			# -----------------------------------------


			# -----------------------------------------
			#/
			# solve for A, o = dense matrix & operator
			#/
			h = lib.pcg_helper_alloc(m, n)

			# form T = (rho I + A'A)
			T = rho * np.eye(n)
			T += self.A_test.dot(self.A_test)

			# allocate linear operator and preconditioner
			A, o = self.gen_dense_operator(dlib, olib, self.A_test)
			p_vec, p = self.gen_preconditioning_operator(dlib, olib, T, rho)

			# test
			lib.pcg_nonallocating(h, o, p, b, x, rho, tol, maxiter, 0)
			dlib.vector_memcpy_av(x_ptr, x, 1)
			x_answer = np.linalg.solve(p * T, self.x_test)
			self.assertTrue(np.linalg.norm(x_answer - x_) <=
							ATOLN + RTOL * np.linalg.norm(x_answer))

			# free components
			self.release_preconditioning_operator(p_vec, p)
			self.release_dense_operator(A, o)

			lib.pcg_helper_free(h)
			# -----------------------------------------

			# -----------------------------------------
			#/
			# solve for A, o = sparse matrix & operator
			#/

			h = lib.pcg_helper_alloc(m, n)

			# form T = (rho I + A'A)
			T = rho * np.eye(n)
			T += self.A_test_sparse.dot(self.A_test_sparse)

			# allocate linear operator and preconditioner
			A, o = self.gen_dense_operator(dlib, olib, self.A_test)
			p_vec, p = self.gen_preconditioning_operator(dlib, olib, T, rho)

			# test
			lib.pcg_nonallocating(h, o, p, b, x, rho, tol, maxiter, 0)
			dlib.vector_memcpy_av(x_ptr, x, 1)
			x_answer = np.linalg.solve(p * T, self.x_test)
			self.assertTrue(np.linalg.norm(x_answer - x_) <=
							ATOLN + RTOL * np.linalg.norm(x_answer))

			lib.pcg_helper_free(h)
			# -----------------------------------------

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)
			# -----------------------------------------

			print "TEST PCG NONALLOCATING"

	def test_pcg_nonallocating_warmstart(self):
		"""TODO: DOCSTRING"""
"""
		tol = 1e-6
		rho = 1e-4
		maxiter = 50

		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5


			# -----------------------------------------
			# allocate x, b in python & C
			b = dlib.vector(0, 0, None)
			dlib.vector_alloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_alloc(b, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			# -----------------------------------------


			# -----------------------------------------
			#/
			# solve for A, o = dense matrix & operator
			#/
			h = lib.pcg_helper_alloc(m, n)

			# form T = (rho I + A'A)
			T = rho * np.eye(n)
			T += self.A_test.dot(self.A_test)

			# allocate linear operator and preconditioner
			A, o = self.gen_dense_operator(dlib, olib, self.A_test)
			p_vec, p = self.gen_preconditioning_operator(dlib, olib, T, rho)

			# test
			lib.pcg_nonallocating(h, o, p, b, x, rho, tol, maxiter, 0)
			dlib.vector_memcpy_av(x_ptr, x, 1)
			x_answer = np.linalg.solve(p * T, self.x_test)
			self.assertTrue(np.linalg.norm(x_answer - x_) <=
							ATOLN + RTOL * np.linalg.norm(x_answer))



			# free components
			self.release_preconditioning_operator(p_vec, p)
			self.release_dense_operator(A, o)

			lib.pcg_helper_free(h)
			# -----------------------------------------

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)
			# -----------------------------------------
"""
			print "TEST PCG NONALLOCATING WARMSTART"


	def test_pcg_allocating(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			print "TEST PCG ALLOCATING"

	def test_pcg_easy(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			dlib = self.dense_libs.get(single_precision=single_precision,
									   gpu=gpu)
			slib = self.sparse_libs.get(dlib,
										single_precision=single_precision,
										gpu=gpu)
			olib = self.op_libs.get(dlib, slib,
								    single_precision=single_precision, gpu=gpu)
			lib = self.cg_libs.get(dlib, olib,
								   single_precision=single_precision, gpu=gpu)

			if lib is None:
				continue

			print "TEST PCG EASY"