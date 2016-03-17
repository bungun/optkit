import unittest
import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref, CFUNCTYPE
from optkit.libs import DenseLinsysLibs, SparseLinsysLibs, OperatorLibs, \
						ConjugateGradientLibs
from optkit.tests.defs import CONDITIONS, DEFAULT_SHAPE, DEFAULT_MATRIX_PATH
import pdb

CG_QUIET = 0

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

	@staticmethod
	def gen_preconditioning_operator(dlib, olib, A_py, rho):
		n = A_py.shape[1]
		p_vec = dlib.vector(0, 0, None)
		dlib.vector_calloc(p_vec, n)
		p_ = np.zeros(n).astype(dlib.pyfloat)
		p_ptr = p_.ctypes.data_as(dlib.ok_float_p)

		# calculate diagonal preconditioner
		for j in xrange(A_py.shape[1]):
			p_[j] = 1. / (rho +  np.linalg.norm(A_py[:, j])**2)

		dlib.vector_memcpy_va(p_vec, p_ptr, 1)
		p = olib.diagonal_operator_alloc(p_vec)
		return p_, p_vec, p

	@staticmethod
	def release_preconditioning_operator(dlib, olib, p_vec, p):
		olib.operator_free(p)
		dlib.vector_free(p_vec)

 	@staticmethod
 	def gen_dense_operator(dlib, olib, A_py, rowmajor=True):
 		m, n = A_py.shape
 		order = dlib.enums.CblasRowMajor if rowmajor else \
 				dlib.enums.CblasColMajor
		pyorder = 'C' if rowmajor else 'F'
		A_ = np.zeros(A_py.shape, order=pyorder).astype(dlib.pyfloat)
		A_ += A_py
		A_ptr = A_.ctypes.data_as(dlib.ok_float_p)
		A = dlib.matrix(0, 0, 0, None, order)
		dlib.matrix_calloc(A, m, n, order)
		dlib.matrix_memcpy_ma(A, A_ptr, order)
		o = olib.dense_operator_alloc(A)
		return A, o

	@staticmethod
	def release_dense_operator(dlib, olib, A, o):
		olib.operator_free(o)
		dlib.matrix_free(A)

	@staticmethod
	def gen_sparse_operator(dlib, slib, olib, A_py, rowmajor=True):
		m, n = A_py.shape
		order = dlib.enums.CblasRowMajor if rowmajor else \
				dlib.enums.CblasColMajor
		sparsemat = csr_matrix if rowmajor else csc_matrix
		sparse_hdl = c_void_p()
		slib.sp_make_handle(byref(sparse_hdl))
		A_ = A_py.astype(dlib.pyfloat)
		A_sp = csr_matrix(A_)
		A_ptr = A_sp.indptr.ctypes.data_as(slib.ok_int_p)
		A_ind = A_sp.indices.ctypes.data_as(slib.ok_int_p)
		A_val = A_sp.data.ctypes.data_as(dlib.ok_float_p)
		A = slib.sparse_matrix(0, 0, 0, 0, None, None, None, order)
		slib.sp_matrix_calloc(A, m, n, A_sp.nnz, order)
		slib.sp_matrix_memcpy_ma(sparse_hdl, A, A_val, A_ind, A_ptr)
		slib.sp_destroy_handle(sparse_hdl)
		o = olib.sparse_operator_alloc(A)
		return A, o

	@staticmethod
	def release_sparse_operator(slib, olib, A, o):
		olib.operator_free(o)
		slib.sp_matrix_free(A)

	@property
	def op_keys(self):
		return ['dense', 'sparse']

	def get_opmethods(self, opkey, denselib, sparselib, operatorlib):
		if opkey == 'dense':
			A = self.A_test
			gen = self.gen_dense_operator
			arg_gen = [denselib, operatorlib, A]
			release = self.release_dense_operator
			arg_release = [denselib, operatorlib]
		elif opkey == 'sparse':
			A = self.A_test_sparse
			gen = self.gen_sparse_operator
			arg_gen = [denselib, sparselib, operatorlib, A]
			release = self.release_sparse_operator
			arg_release = [sparselib, operatorlib]
		else:
			raise ValueError('invalid operator type')

		return (A, gen, arg_gen, release, arg_release)

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

			# -----------------------------------------
			# allocate x, b in python & C
			b = dlib.vector(0, 0, None)
			dlib.vector_calloc(b, m)
			b_ = np.zeros(m).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += np.random.rand(m)
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (nonallocating), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				h = lib.cgls_helper_alloc(m, n)

				A, o = gen_operator(*gen_args)

				flag = lib.cgls_nonallocating(h, o, b, x, rho, tol, maxiter,
											  CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)

				# checks:
				# 1. exit flag == 0
				# 2. KKT condition A'(Ax - b) + rho (x) == 0 (within tol)
				self.assertEqual(flag, 0)
				KKT = A_.T.dot(A_.dot(x_) - b_) + rho * x_
				self.assertTrue(np.linalg.norm(KKT) <= (tol * n)**0.5)

				release_args += [A, o]
				release_operator(*release_args)

				lib.cgls_helper_free(h)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

	def test_cgls_allocating(self):
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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

			# -----------------------------------------
			# allocate x, b in python & C
			b = dlib.vector(0, 0, None)
			dlib.vector_calloc(b, m)
			b_ = np.zeros(m).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += np.random.rand(m)
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (allocating), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				flag = lib.cgls(o, b, x, rho, tol, maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)

				A, o = gen_operator(*gen_args)

				# checks:
				# 1. exit flag == 0
				# 2. KKT condition A'(Ax - b) + rho (x) == 0 (within tol)
				self.assertEqual(flag, 0)
				KKT = A_.T.dot(A_.dot(x_) - b_) + rho * x_
				self.assertTrue(np.linalg.norm(KKT) <= (tol * n)**0.5)

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

	def test_cgls_easy(self):
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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

			# -----------------------------------------
			# allocate x, b in python & C
			b = dlib.vector(0, 0, None)
			dlib.vector_calloc(b, m)
			b_ = np.zeros(m).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += np.random.rand(m)
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (easy), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				A, o = gen_operator(*gen_args)

				cgls_work = lib.cgls_easy_init(m, n)
				flag = lib.cgls_easy_solve(cgls_work, o, b, x, rho, tol,
										   maxiter, CG_QUIET)
				lib.cgls_easy_finish(cgls_work)
				dlib.vector_memcpy_av(x_ptr, x, 1)

				# checks:
				# 1. exit flag == 0
				# 2. KKT condition A'(Ax - b) + rho (x) == 0 (within tol)
				self.assertEqual(flag, 0)
				KKT = A_.T.dot(A_.dot(x_) - b_) + rho * x_
				self.assertTrue(np.linalg.norm(KKT) <= (tol * n)**0.5)

				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

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

	def test_diagonal_preconditioner(self):
		tol = self.tol_cg
		rho = 1e-2
		# rho = self.rho_cg
		maxiter = self.maxiter_cg

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

			RTOL = 2e-2
			ATOLN = RTOL * n**0.5

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				p_vec = dlib.vector(0, 0, None)
				dlib.vector_calloc(p_vec, n)
				p_ = np.zeros(n).astype(dlib.pyfloat)
				p_py = np.zeros(n).astype(dlib.pyfloat)
				p_ptr = p_.ctypes.data_as(dlib.ok_float_p)

				# calculate diagonal preconditioner
				for j in xrange(n):
					p_py[j] = 1. / (rho +  np.linalg.norm(T[:, j])**2)

				A, o = gen_operator(*gen_args)
				lib.diagonal_preconditioner(o, p_vec, rho)
				dlib.vector_memcpy_av(p_ptr, p_vec, 1)
				self.assertTrue(np.linalg.norm(p_py - p_) <=
								ATOLN + RTOL * np.linalg.norm(p_py))

				release_args += [A, o]
				release_operator(*release_args)

	def test_pcg_nonallocating(self):
		"""
		pcg_nonallocating test

		given operator A, vector b, preconditioner M and scalar rho,
		pcg method attemps to solve

			(rho * I + A'A)x = b

		to specified tolerance _tol_ by performing at most _maxiter_
		CG iterations on the system

			M(rho * I + A'A)x = b
		"""
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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
			dlib.vector_calloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				h = lib.pcg_helper_alloc(m, n)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				A, o = gen_operator(*gen_args)
				p_py, p_vec, p = self.gen_preconditioning_operator(dlib, olib,
																   T, rho)

				lib.pcg_nonallocating(h, o, p, b, x, rho, tol, maxiter,
									  CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				self.release_preconditioning_operator(dlib, olib, p_vec, p)
				release_args += [A, o]
				release_operator(*release_args)

				lib.pcg_helper_free(h)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

	def test_pcg_nonallocating_warmstart(self):
		"""TODO: DOCSTRING"""
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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
			dlib.vector_calloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating) warmstart, operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				h = lib.pcg_helper_alloc(m, n)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				A, o = gen_operator(*gen_args)
				p_py, p_vec, p = self.gen_preconditioning_operator(dlib, olib,
																   T, rho)

				iters1 = lib.pcg_nonallocating(h, o, p, b, x, rho, tol,
											   maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				iters2 = lib.pcg_nonallocating(h, o, p, b, x, rho, tol,
											   maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				print 'cold start iters:', iters1
				print 'warm start iters:', iters2
				self.assertTrue(iters2 <= iters1)

				self.release_preconditioning_operator(dlib, olib, p_vec, p)
				release_args += [A, o]
				release_operator(*release_args)

				lib.pcg_helper_free(h)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

	def test_pcg_allocating(self):
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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
			dlib.vector_calloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (allocating), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				A, o = gen_operator(*gen_args)
				p_py, p_vec, p = self.gen_preconditioning_operator(dlib, olib,
																   T, rho)

				lib.pcg(o, p, b, x, rho, tol, maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				self.release_preconditioning_operator(dlib, olib, p_vec, p)
				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)

	def test_pcg_easy(self):
		tol = self.tol_cg
		rho = self.rho_cg
		maxiter = self.maxiter_cg

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
			dlib.vector_calloc(b, n)
			b_ = np.zeros(n).astype(dlib.pyfloat)
			b_ptr = b_.ctypes.data_as(dlib.ok_float_p)

			x = dlib.vector(0, 0, None)
			dlib.vector_calloc(x, n)
			x_ = np.zeros(n).astype(dlib.pyfloat)
			x_ptr = x_.ctypes.data_as(dlib.ok_float_p)

			b_ += self.x_test
			dlib.vector_memcpy_va(b, b_ptr, 1)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (easy), operator type:", op_
				(
						A_, gen_operator, gen_args, release_operator,
						release_args
				) = self.get_opmethods(op_, dlib, slib, olib)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				A, o = gen_operator(*gen_args)
				p_py, p_vec, p = self.gen_preconditioning_operator(dlib, olib,
																   T, rho)

				pcg_work = lib.pcg_easy_init(m, n)
				iters1 = lib.pcg_easy_solve(pcg_work, o, p, b, x, rho, tol,
											maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				iters2 = lib.pcg_easy_solve(pcg_work, o, p, b, x, rho, tol,
											maxiter, CG_QUIET)
				dlib.vector_memcpy_av(x_ptr, x, 1)
				self.assertTrue(np.linalg.norm(T.dot(x_) - b_) <=
								ATOLN + RTOL * np.linalg.norm(b_))

				print 'cold start iters:', iters1
				print 'warm start iters:', iters2
				self.assertTrue(iters2 <= iters1)

				lib.pcg_easy_finish(pcg_work)
				self.release_preconditioning_operator(dlib, olib, p_vec, p)
				release_args += [A, o]
				release_operator(*release_args)

			# -----------------------------------------
			# free x, b
			dlib.vector_free(x)
			dlib.vector_free(b)