import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs.cg import ConjugateGradientLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import optkit.tests.C.operator_helper as op_helper
from optkit.tests.C.base import OptkitCTestCase

CG_QUIET = 0

class ConjugateGradientLibsTestCase(OptkitCTestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ConjugateGradientLibs()

		self.tol_cg = 1e-12
		self.rho_cg = 1e-4
		self.maxiter_cg = 1000

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
		self.nnz = sum(sum(self.A_test_sparse > 0))

	def tearDown(self):
		self.free_all_vars()

	@staticmethod
	def gen_preconditioning_operator(lib, A_py, rho):
		n = A_py.shape[1]
		p_vec = lib.vector(0, 0, None)
		lib.vector_calloc(p_vec, n)
		p_ = np.zeros(n).astype(lib.pyfloat)
		p_ptr = p_.ctypes.data_as(lib.ok_float_p)

		# calculate diagonal preconditioner
		for j in xrange(A_py.shape[1]):
			p_[j] = 1. / (rho +  np.linalg.norm(A_py[:, j])**2)

		lib.vector_memcpy_va(p_vec, p_ptr, 1)
		p = lib.diagonal_operator_alloc(p_vec)
		return p_, p_vec, p, lib.vector_free

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def gen_operator(self, opkey, lib):
		if opkey == 'dense':
			return op_helper.gen_dense_operator(lib, self.A_test)
		elif opkey == 'sparse':
			return op_helper.gen_dense_operator(lib, self.A_test_sparse)
		else:
			raise ValueError('invalid operator type')

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

	def test_cgls_helper_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)			if lib is None:
				continue

			h = lib.cgls_helper_alloc(self.shape[0], self.shape[1])
			self.assertTrue(isinstance(h.contents.p, lib.vector_p))
			self.assertTrue(isinstance(h.contents.q, lib.vector_p))
			self.assertTrue(isinstance(h.contents.r, lib.vector_p))
			self.assertTrue(isinstance(h.contents.s, lib.vector_p))
			lib.cgls_helper_free(h)

	def test_cgls_nonallocating(self):
		"""
		cgls_nonallocating test

		given operator A, vector b and scalar rho,
		cgls method attemps to solve

			min. ||Ax - b||_2^2 + rho ||x||_2^2

		to specified tolerance _tol_ by performing at most _maxiter_
		CG iterations on the above least squares problem
		"""
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			DIGITS = 7 - 2 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5

			# -----------------------------------------
			# allocate x, b in python & C
			b = lib.vector(0, 0, None)
			lib.vector_calloc(b, n)
			self.register_var('b', b, lib.vector_free)
			b_ = np.zeros(n).astype(lib.pyfloat)
			b_ptr = b_.ctypes.data_as(lib.ok_float_p)

			x = lib.vector(0, 0, None)
			lib.vector_calloc(x, n)
			self.register_var('x', x, lib.vector_free)
			x_ = np.zeros(n).astype(lib.pyfloat)
			x_ptr = x_.ctypes.data_as(lib.ok_float_p)

			b_ += self.x_test
			lib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (easy), operator type:", op_
				A, o, freeA = self.gen_operator(op_, lib)
				self.register_var('A', A, freeA)
				self.register_var('o', o, o.contents.free)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				p_py, p_vec, p, free_p = self.gen_preconditioning_operator(
						lib, T, rho)
				self.register_var('p_vec', p_vec, free_p)
				self.register_var('p', p, p.contents.free)

				pcg_work = lib.pcg_init(m, n)
				self.register_var('work', pcg_work, lib.pcg_finish)
				iters1 = lib.pcg_solve(pcg_work, o, p, b, x, rho, tol,
											maxiter, CG_QUIET)
				lib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				iters2 = lib.pcg_solve(pcg_work, o, p, b, x, rho, tol,
											maxiter, CG_QUIET)
				lib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				print 'cold start iters:', iters1
				print 'warm start iters:', iters2
				self.assertTrue(iters2 <= iters1)

				self.free_var('work')
				self.free_var('p')
				self.free_var('p_vec')
				self.free_var('o')
				self.free_var('A')

			# -----------------------------------------
			# free x, b
			self.free_var('x')
			self.free_var('b')