import os
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_uint, c_void_p, byref, POINTER
from optkit.libs.cg import ConjugateGradientLibs
from optkit.tests.C.base import OptkitCTestCase, OptkitCOperatorTestCase

CG_QUIET = 1

class ConjugateGradientLibsTestCase(OptkitCOperatorTestCase):
	"""TODO: docstring"""

	@classmethod
	def setUpClass(self):
		self.env_orig = os.getenv('OPTKIT_USE_LOCALLIBS', '0')
		os.environ['OPTKIT_USE_LOCALLIBS'] = '1'
		self.libs = ConjugateGradientLibs()

		self.tol_cg = 1e-12
		self.rho_cg = 1e-4
		self.maxiter_cg = 1000

		self.A_test = self.A_test_gen
		self.A_test_sparse = self.A_test_sparse_gen

	@classmethod
	def tearDownClass(self):
		os.environ['OPTKIT_USE_LOCALLIBS'] = self.env_orig

	def setUp(self):
		self.x_test = np.random.rand(self.shape[1])

	def tearDown(self):
		self.free_all_vars()
		self.exit_call()

	def register_preconditioning_operator(self, lib, A_py, rho):
		n = A_py.shape[1]
		p_vec, p_, p_ptr = self.register_vector(lib, n, 'p_vec')

		# calculate diagonal preconditioner
		for j in xrange(A_py.shape[1]):
			p_[j] = 1. / (rho +  np.linalg.norm(A_py[:, j])**2)

		self.assertCall( lib.vector_memcpy_va(p_vec, p_ptr, 1) )
		p = lib.diagonal_operator_alloc(p_vec)
		self.register_var('p', p.contents.data, p.contents.free)
		return p_, p_vec, p

	def test_libs_exist(self):
		libs = []
		for (gpu, single_precision) in self.CONDITIONS:
			libs.append(self.libs.get(single_precision=single_precision,
									  gpu=gpu))
		self.assertTrue(any(libs))

	def assert_cgls_exit(self, A, x, b, rho, flag, tol):
		# checks:
		# 1. exit flag == 0
		# 2. KKT condition A'(Ax - b) + rho (x) == 0 (within tol)
		self.assertTrue( flag == 0 )
		KKT = A.T.dot(A.dot(x) - b) + rho * x
		self.assertTrue( np.linalg.norm(KKT) <= tol )

	def test_cgls_helper_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			h = lib.cgls_helper_alloc(self.shape[0], self.shape[1])
			self.register_var('h', h, lib.cgls_helper_free)
			self.assertTrue( isinstance(h.contents.p, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.q, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.r, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.s, lib.vector_p) )
			self.assertCall( lib.cgls_helper_free(h) )
			self.unregister_var('h')

	def test_cgls_nonallocating(self):
		"""
		cgls_nonallocating test

		given operator A, vector b and scalar rho,
		cgls method attemps to solve

			min. ||Ax - b||_2^2 + rho ||x||_2^2

		to specified tolerance _tol_ by performing at most _maxiter_
		CG iterations on the above least squares problem
		"""
		m, n = self.shape
		tol = self.tol_cg
		maxiter = self.maxiter_cg
		rho = 1e-2

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			ATOLN = n**0.5 * 10**(-7 + 3 * single_precision)
			TOL = tol * 10**(5 * single_precision)
			RHO = rho * 10**(1 * single_precision)

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (nonallocating), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, m, 'b')

				b_ += np.random.rand(m)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A_, A, o = self.register_operator(lib, op_)

				h = lib.cgls_helper_alloc(m, n)
				self.register_var('h', h, lib.cgls_helper_free)

				flag = np.zeros(1).astype(c_uint)
				flag_p = flag.ctypes.data_as(POINTER(c_uint))
				self.assertCall( lib.cgls_nonallocating(h, o, b, x, RHO, TOL,
														maxiter, CG_QUIET,
														flag_p) )
				self.assertTrue( flag[0] <= lib.CGLS_MAXFLAG )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assert_cgls_exit(A_, x_, b_, RHO, flag[0], ATOLN)

				self.free_vars('o', 'A', 'h', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_cgls_allocating(self):
		tol = self.tol_cg
		maxiter = self.maxiter_cg
		rho = 1e-2

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			ATOLN = n**0.5 * 10**(-7 + 3 * single_precision)
			TOL = tol * 10**(5 * single_precision)
			RHO = rho * 10**(1 * single_precision)

			# -----------------------------------------
			# allocate x, b in python & C
			x, x_, x_ptr = self.register_vector(lib, n, 'x')
			b, b_, b_ptr = self.register_vector(lib, m, 'b')

			b_ += np.random.rand(m)
			self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (allocating), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, m, 'b')

				b_ += np.random.rand(m)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A_, A, o = self.register_operator(lib, op_)

				flag = np.zeros(1).astype(c_uint)
				flag_p = flag.ctypes.data_as(POINTER(c_uint))

				self.assertCall( lib.cgls(o, b, x, RHO, TOL, maxiter,
										  CG_QUIET, flag_p) )
				self.assertTrue( flag[0] <= lib.CGLS_MAXFLAG )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assert_cgls_exit(A_, x_, b_, RHO, flag[0], ATOLN)

				self.free_vars('o', 'A', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_cgls_easy(self):
		tol = self.tol_cg
		maxiter = self.maxiter_cg
		rho = 1e-2

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			ATOLN = n**0.5 * 10**(-7 + 3 * single_precision)
			TOL = tol * 10**(5 * single_precision)
			RHO = rho * 10**(1 * single_precision)

			# -----------------------------------------
			# test cgls for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test cgls (easy), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, m, 'b')

				b_ += np.random.rand(m)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A_, A, o = self.register_operator(lib, op_)

				cgls_work = lib.cgls_init(m, n)
				self.register_var('work', cgls_work, lib.cgls_finish)

				flag = np.zeros(1).astype(c_uint)
				flag_p = flag.ctypes.data_as(POINTER(c_uint))

				self.assertCall( lib.cgls_solve(cgls_work, o, b, x, RHO,
												TOL, maxiter, CG_QUIET,
												flag_p) )
				self.assertTrue( flag[0] <= lib.CGLS_MAXFLAG )
				self.free_var('work')
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )

				self.assert_cgls_exit(A_, x_, b_, RHO, flag[0], ATOLN)
				self.free_vars('o', 'A', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_pcg_helper_alloc_free(self):
		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue

			h = lib.pcg_helper_alloc(self.shape[0], self.shape[1])
			self.register_var('h', h, lib.pcg_helper_free)
			self.assertTrue( isinstance(h.contents.p, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.q, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.r, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.z, lib.vector_p) )
			self.assertTrue( isinstance(h.contents.temp, lib.vector_p) )
			self.assertCall( lib.pcg_helper_free(h) )
			self.unregister_var('h')

	def test_diagonal_preconditioner(self):
		tol = self.tol_cg
		rho = 1e-2
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			RTOL = 2e-2
			ATOLN = RTOL * n**0.5

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating), operator type:", op_
				A_, A, o = self.register_operator(lib, op_)

				T = rho * np.eye(n)
				T += A_.T.dot(A_)

				p_vec, p_, p_ptr = self.register_vector(lib, n, 'p_vec')

				# calculate diagonal preconditioner
				p_py = np.zeros(n)
				for j in xrange(n):
					p_py[j] = 1. / (rho +  np.linalg.norm(T[:, j])**2)

				self.assertCall( lib.diagonal_preconditioner(o, p_vec, rho) )
				self.assertCall( lib.vector_memcpy_av(p_ptr, p_vec, 1) )
				self.assertVecEqual( p_py, p_, ATOLN, RTOL )

				self.free_vars('o', 'A', 'p_vec')
				self.assertCall( lib.ok_device_reset() )

	def register_pcg_operators(self, lib, optype, rho, n):
		A_, A, o = self.register_operator(lib, optype)
		T = rho * np.eye(n)
		T += A_.T.dot(A_)
		p_py, p_vec, p = self.register_preconditioning_operator(lib, T, rho)

		return A, o, T, p_vec, p

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
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			RHO = self.rho_cg * 10**(4 * single_precision)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, n, 'b')

				b_ += np.random.rand(n)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A, o, T, p_vec, p = self.register_pcg_operators(lib, op_, RHO,
																n)

				h = lib.pcg_helper_alloc(m, n)
				self.register_var('h', h, lib.pcg_helper_free)

				iter_ = np.zeros(1).astype(c_uint)
				iter_p = iter_.ctypes.data_as(POINTER(c_uint))
				self.assertCall( lib.pcg_nonallocating(
						h, o, p, b, x, RHO, tol, maxiter, CG_QUIET, iter_p) )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertTrue( iter_[0] <= maxiter )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				self.free_vars('p', 'p_vec', 'o', 'A', 'h', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_pcg_nonallocating_warmstart(self):
		"""TODO: DOCSTRING"""
		tol = self.tol_cg
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			RHO = self.rho_cg * 10**(4 * single_precision)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (nonallocating) warmstart, operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, n, 'b')

				b_ += np.random.rand(n)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A, o, T, p_vec, p = self.register_pcg_operators(lib, op_, RHO,
																n)

				h = lib.pcg_helper_alloc(m, n)
				self.register_var('h', h, lib.pcg_helper_free)

				iter_ = np.zeros(1).astype(c_uint)
				iter_p = iter_.ctypes.data_as(POINTER(c_uint))

				# first run
				self.assertCall( lib.pcg_nonallocating(
						h, o, p, b, x, RHO, tol, maxiter, CG_QUIET, iter_p) )
				iters1 = iter_[0]
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				# second run
				self.assertCall( lib.pcg_nonallocating(
						h, o, p, b, x, RHO, tol, maxiter, CG_QUIET, iter_p) )
				iters2 = iter_[0]
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				print 'cold start iters:', iters1
				print 'warm start iters:', iters2
				self.assertTrue(iters1 <= maxiter)
				self.assertTrue(iters2 <= maxiter)
				self.assertTrue(iters2 <= iters1)

				self.free_vars('p', 'p_vec', 'o', 'A', 'h', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_pcg_allocating(self):
		tol = self.tol_cg
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			RHO = self.rho_cg * 10**(4 * single_precision)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (allocating), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, n, 'b')

				b_ += np.random.rand(n)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A, o, T, p_vec, p = self.register_pcg_operators(lib, op_, RHO,
																n)

				h = lib.pcg_helper_alloc(m, n)
				self.register_var('h', h, lib.pcg_helper_free)

				iter_ = np.zeros(1).astype(c_uint)
				iter_p = iter_.ctypes.data_as(POINTER(c_uint))

				self.assertCall( lib.pcg(o, p, b, x, RHO, tol, maxiter,
					CG_QUIET, iter_p) )
				self.assertTrue( iter_[0] <= maxiter )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				self.free_vars('h', 'p', 'p_vec', 'o', 'A', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )

	def test_pcg_easy(self):
		tol = self.tol_cg
		maxiter = self.maxiter_cg

		m, n = self.shape

		for (gpu, single_precision) in self.CONDITIONS:
			lib = self.libs.get(single_precision=single_precision, gpu=gpu)
			if lib is None:
				continue
			self.register_exit(lib.ok_device_reset)

			DIGITS = 7 - 5 * single_precision - 1 * gpu
			RTOL = 10**(-DIGITS)
			ATOLN = RTOL * n**0.5
			RHO = self.rho_cg * 10**(4 * single_precision)

			# -----------------------------------------
			# test pcg for each operator type defined in self.op_keys
			for op_ in self.op_keys:
				print "test pcg (easy), operator type:", op_
				x, x_, x_ptr = self.register_vector(lib, n, 'x')
				b, b_, b_ptr = self.register_vector(lib, n, 'b')

				b_ += np.random.rand(n)
				self.assertCall( lib.vector_memcpy_va(b, b_ptr, 1) )

				A, o, T, p_vec, p = self.register_pcg_operators(lib, op_, RHO,
																n)
				h = lib.pcg_helper_alloc(m, n)
				self.register_var('h', h, lib.pcg_helper_free)

				pcg_work = lib.pcg_init(m, n)
				self.register_var('work', pcg_work, lib.pcg_finish)

				iter_ = np.zeros(1).astype(c_uint)
				iter_p = iter_.ctypes.data_as(POINTER(c_uint))

				self.assertCall( lib.pcg_solve(pcg_work, o, p, b, x, RHO, tol,
											   maxiter, CG_QUIET, iter_p) )
				iters1 = iter_[0]
				self.assertTrue( iters1 <= maxiter )
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				self.assertCall( lib.pcg_solve(pcg_work, o, p, b, x, RHO, tol,
											   maxiter, CG_QUIET, iter_p) )
				iters2 = iter_[0]
				self.assertTrue(iters2 <= maxiter)
				self.assertCall( lib.vector_memcpy_av(x_ptr, x, 1) )
				self.assertVecEqual( T.dot(x_), b_, ATOLN, RTOL )

				print 'cold start iters:', iters1
				print 'warm start iters:', iters2
				self.assertTrue(iters2 <= iters1)

				self.free_vars('work', 'h', 'p', 'p_vec', 'o', 'A', 'x', 'b')
				self.assertCall( lib.ok_device_reset() )