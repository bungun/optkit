from traceback import format_exc
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix
from ctypes import c_void_p, byref
from optkit.tests.defs import gen_test_defs
from optkit.types.lowlevel import ok_enums
from optkit.utils.pyutils import println, printvoid, \
	var_assert, array_compare, pretty_print
import pdb 

def test_lowlevelvectorcalls(errors, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		make_cvector = backend.make_cvector

		blas_handle = c_void_p()
		backend.dense.blas_make_handle(byref(blas_handle))

		v = backend.make_cvector()
		w = backend.make_cvector()
		backend.dense.vector_calloc(v, 10)
		backend.dense.vector_calloc(w, 10)
		backend.dense.vector_add_constant(v, 12.)
		backend.dense.vector_add_constant(w, 5.)
		backend.dense.blas_axpy(blas_handle, 1., v , w)
		if VERBOSE_TEST:
			backend.dense.vector_print(v)
			backend.dense.vector_print(w)
		backend.dense.vector_free(v)
		backend.dense.vector_free(w)
		return True

	except:
		errors.append(format_exc())
		return False

def test_lowlevelmatrixcalls(errors, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		A = backend.make_cmatrix()
		backend.dense.matrix_calloc(A, 10,  10, ok_enums.CblasRowMajor)
		if VERBOSE_TEST:
			backend.dense.matrix_print(A)
		backend.dense.matrix_free(A)
		return True
		
	except:
		errors.append(format_exc())
		return False

def test_lowlevelsparsecalls(errors, VERBOSE_TEST=True,
	gpu=False, floatbits=64):
	
	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		make_csparsematrix = backend.make_csparsematrix

		assert backend.sparse is not None

		sparse_handle = c_void_p(0)

		backend.sparse.sp_make_handle(byref(sparse_handle))
		A = backend.make_csparsematrix()
		backend.sparse.sp_matrix_calloc(A, 10,  10, 22, ok_enums.CblasRowMajor)
		if VERBOSE_TEST:
			backend.sparse.sp_matrix_print(A)
		backend.sparse.sp_matrix_free(A)

		backend.sparse.sp_destroy_handle(sparse_handle)
		return True

	except:
		errors.append(format_exc())
		return False



def test_vector_methods(errors, n=3,VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import linsys, Vector, Matrix
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)


		if n is None: n = 3
		if n < 3: 
			print str("length provided: {}.\n"
			"using 3 as minimum vector length for tests".format(n))
			n=3

		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid
		PRINTVAR = linsys['print_var'] if VERBOSE_TEST else printvoid
		VEC_ASSERT = lambda *v : var_assert(*v, type=Vector)


		PPRINT("VECTOR METHODS")


		PRINT("\nAllocation (int size={}) & printing".format(n))
		PRINT("---variable `a`---")
		a=Vector(n)
		assert VEC_ASSERT(a)
		a_ = np.copy(a.py)


		PRINT("numpy value:")
		PRINT(a.py)
		PRINT("c value:")
		PRINTVAR(a)

		PRINT("\nSync")
		PRINT("c to python")

		orig_pointer = a.py.ctypes._as_parameter_.value
		linsys['sync'](a)
		assert a.py.ctypes._as_parameter_.value == orig_pointer
		PRINT(a.py)
		PRINTVAR(a)

		PRINT("python to c")
		orig_pointer = a.c.data
		linsys['sync'](a,python_to_C=1)
		# assert a.c.data == orig_pointer
		PRINT(a.py)
		PRINTVAR(a)


		PRINT("\nConstant addition: a += 1")
		linsys['add'](1,a)
		linsys['sync'](a)
		a_ += 1
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)

		PRINT("\nScalar multiplication: a *= 5")
		linsys['mul'](5,a)
		linsys['sync'](a)
		a_ *= 5
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)

		PRINT("\nScalar division: a /= 2.3")
		linsys['div'](2.3,a)
		linsys['sync'](a)
		a_ /= 2.3
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)

		b_ = RAND_ARR(n)
		PRINT("\nAllocation (from numpy array {})".format(b_))
		PRINT("---variable `b`---")
		b=Vector(np.copy(b_))
		PRINTVAR(b)
		assert VEC_ASSERT(b)
		assert array_compare(b.py,b_, eps=TEST_EPS)


		PRINT("\nElementwise vector addition: a += b")
		linsys['add'](b,a)
		linsys['sync'](a,b)
		a_ += b_
		PRINT("a:")
		PRINTVAR(a)
		PRINT("b:")
		PRINTVAR(b)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)


		PRINT("\nElementwise vector multiplication: a *= b")
		linsys['mul'](b,a)
		linsys['sync'](a,b)
		a_ *= b_
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)


		PRINT("\nElementwise vector division: a /= b")
		linsys['div'](b,a)
		linsys['sync'](a,b)
		a_ /= b_
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)


		PRINT("\nElementwise vector subtraction: a-= b")
		linsys['sub'](b,a)
		linsys['sync'](a,b)
		a_ -= b_
		PRINTVAR(a)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)


		PRINT("\nView")
		PRINT("---variable `a_view`---")
		PRINT("a_view = view(a, 0:2)")
		c=linsys['view'](a,(0,2))
		linsys['sync'](c)
		c_ = a_[0:2]
		PRINTVAR(c)
		assert array_compare(c.py,c_, eps=TEST_EPS)

		PRINT("a_view *=2")
		linsys['mul'](2,c)
		linsys['sync'](c)
		c_ *=2

		PRINT("a_view:")
		PRINTVAR(c)
		PRINT("a:")
		PRINTVAR(a)	
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(c.py,c_, eps=TEST_EPS)


		PRINT("Set all")
		val = 0.2
		linsys['set_all'](val, a)
		linsys['sync'](a)
		PRINTVAR(a)
		PRINT(a.py)
		assert all(a.py == 0.2)

		return True

	except:
		errors.append(format_exc())
		return False

def test_matrix_methods(errors, m=4, n=3, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import linsys, Vector, Matrix
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		if m is None: m=4
		if n is None: n=3
		if n < 3 or m < 3: 
			print str("shape provided: ({},{}).\n"
			"using (3,3) as minimum matrix size for tests".format(m,n))
			n=max(n,3)
			m=max(m,3)

		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid
		PRINTVAR = linsys['print_var'] if VERBOSE_TEST else printvoid
		VEC_ASSERT = lambda *v : var_assert(*v, type=Vector)
		MAT_ASSERT = lambda *A : var_assert(*A, type=Matrix)

		PPRINT("MATRIX METHODS")


		PRINT("\nAllocation (int size1={},size2={}) & printing".format(m,n))
		PRINT("---variable `A`---")
		# matrix
		A=Matrix(m,n)
		assert MAT_ASSERT(A)
		# python clone
		A_ = np.copy(A.py)

		PRINT("numpy value:")
		PRINT(A.py)
		PRINT("c value:")
		PRINTVAR(A)

		PRINT("\nSync")

		PRINT("c to python")
		PRINT("A += 1 (Python)")
		# modify python value
		A.py += 1
		# modify clone to match change to data in CPU case
		if not A.on_gpu: A_ += 1

		# load C value to python (expect overwrite in GPU case)
		orig_pointer = A.py.ctypes._as_parameter_.value
		linsys['sync'](A)
		PRINT(A.py)
		PRINTVAR(A)
		assert A.py.ctypes._as_parameter_.value == orig_pointer
		assert array_compare(A.py,A_, eps=TEST_EPS)

		PRINT("python to c")
		PRINT("A += 1 (Python)")
		# modify python value and clone
		A.py+=1
		A_ += 1

		# orig_pointer = A.c.data.value
		# send python value to C
		linsys['sync'](A,python_to_C=1)
		# assert A.c.data.value == orig_pointer
		PRINT(A.py)
		PRINTVAR(A)
		# load C value back to python, compare to clone
		linsys['sync'](A)
		assert array_compare(A.py,A_, eps=TEST_EPS)

		PRINT("Allocation from row-major numpy ndarray")
		A2_ = np.ndarray(shape=(m,n),order='C')
		for i in xrange(m*n):
			A2_.itemset(i,i)
		A2 = Matrix(A2_)
		assert array_compare(A2.py, A2_, eps=TEST_EPS)

		PRINT("Allocation from column-major numpy ndarray")
		A3_ = np.ndarray(shape=(m,n),order='F')
		for i in xrange(m*n):
			A3_.itemset(i,i)
		A3 = Matrix(A3_)
		assert array_compare(A3.py, A3_, eps=TEST_EPS)

		PRINT("\nMatrix views:")

		PRINT("\n---variable `a_col`---")
		PRINT("column view: A[1]:")
		a_col = linsys['view'](A,1,col=1)
		assert VEC_ASSERT(a_col)
		assert a_col.size == A.size1
		ac_ = A_[:,1]
		assert array_compare(a_col.py,ac_, eps=TEST_EPS)

		PRINT("a_col +=3.5")
		linsys['add'](3.5,a_col)
		linsys['sync'](a_col) 	
		ac_ += 3.5

		PRINT("a_col:")
		PRINTVAR(a_col)
		PRINT(a_col.py)
		PRINT("A:")
		PRINTVAR(A)
		PRINT(A.py)

		assert array_compare(a_col.py, ac_, eps=TEST_EPS)
		assert array_compare(A.py,A_, eps=TEST_EPS)


		PRINT("\n---variable `a_row`---")
		PRINT("row view: A[1]:")
		a_row = linsys['view'](A,1,row=1)
		assert VEC_ASSERT(a_row)
		assert a_row.size == A.size2
		ar_ = A_[1,:]
		assert array_compare(a_row.py,ar_, eps=TEST_EPS)

		PRINT("a_row -=5.34")
		linsys['sub'](5.34,a_row)
		linsys['sync'](a_row)
		ar_ -= 5.34

		PRINT("a_row:")
		PRINTVAR(a_row)
		PRINT(a_row.py)
		PRINT("A:")
		PRINTVAR(A)
		PRINT(A.py)
		assert array_compare(a_row.py,ar_, eps=TEST_EPS)
		assert array_compare(A.py,A_, eps=TEST_EPS)


		PRINT("\n---variable `a_diag'---")
		PRINT("diag view: A:")
		a_diag = linsys['view'](A,diag=1)
		assert VEC_ASSERT(a_diag)
		assert a_diag.size == A.mindim
		ad_ = np.diag(A_).copy()
		assert array_compare(a_diag.py,ad_, eps=TEST_EPS)

		PRINT("a_diag /=2.1")
		linsys['div'](2.1,a_diag)
		ad_ /= 2.1
		PRINT("a_diag:")
		PRINTVAR(a_diag)
		PRINT(a_diag.py)
		PRINT("A:")
		PRINTVAR(A)
		PRINT(A.py)
		assert all(ad_ == 0) or not array_compare(a_diag.py,ad_, 
			eps=TEST_EPS, expect=False)
		assert all(ad_ == 0) or A.sync_required or not array_compare(A.py,A_, 
			eps=TEST_EPS, expect=False)


		PRINT("\nsync diagonal")
		linsys['sync'](a_diag, A)
		for i in xrange(min(m,n)):
			A_[i,i] /= 2.1

		PRINT("a_diag:")
		PRINTVAR(a_diag)
		PRINT(a_diag.py)
		assert array_compare(a_diag.py,ad_, eps=TEST_EPS)
		assert array_compare(A.py,A_, eps=TEST_EPS)


		PRINT("\nsubmatrix")
		PRINT("\n---variable `a_sub`---")
		PRINT("view: A[0:1,1:2]:")
		a_sub = linsys['view'](A,(0,1),(1,2))
		assert MAT_ASSERT(a_sub)
		as_ = A_[0:1,1:2]
		assert array_compare(a_sub.py,as_, eps=TEST_EPS)


		PRINT("a_sub /=3.5")
		linsys['div'](3.5,a_sub)
		linsys['sync'](a_sub)
		as_ /= 3.5

		PRINT("a_sub:")
		PRINTVAR(a_sub)
		PRINT(a_sub.py)
		PRINT("A:")
		PRINTVAR(A)
		PRINT(A.py)
		assert array_compare(a_sub.py,as_, eps=TEST_EPS)
		assert array_compare(A.py,A_, eps=TEST_EPS)


		PRINT("\nAllocation (from {} by {} RAND_ARR array)".format(n,n))
		PRINT("---variable `B`---")
		B=Matrix(RAND_ARR(n,n))
		assert MAT_ASSERT(B)
		B_ = np.copy(B.py)


		PRINT("numpy value:")
		PRINT(B.py)
		PRINT("c value:")
		PRINTVAR(B)

		PRINT("\nMatrix scaling")
		PRINT("B *= 2")
		linsys['mul'](2.,B)
		linsys['sync'](B)
		B_ *= 2.

		PRINT(B.py)
		PRINTVAR(B)
		assert array_compare(B.py,B_, eps=TEST_EPS)


		PRINT("\nAllocation (from {}x{} RAND_ARR array)".format(m,n))
		PRINT("---variable `C`---")
		C=Matrix(RAND_ARR(m,n))
		assert MAT_ASSERT(C)

		PRINT(C.py)
		PRINTVAR(C)


		PRINT("\nMatrix->Matrix Copy")
		PRINT("---variable `D` = 0^({}x{})---".format(n,n))
		D=Matrix(np.zeros_like(B.py))
		assert MAT_ASSERT(D)
		D_ = np.copy(D.py)

		PRINT(D.py)
		PRINTVAR(D)

		PRINT("D := memcopy(B)")
		linsys['copy'](B,D)
		linsys['sync'](B,D)
		D_[:]=B_[:]

		PRINT(D.py)
		PRINTVAR(D)
		assert array_compare(B.py,D.py, eps=TEST_EPS)
		assert array_compare(B.py,B_, eps=TEST_EPS)
		assert array_compare(D.py,D_, eps=TEST_EPS)


		PRINT("\nndarray->Matrix Copy")

		PRINT("Copy: row-major numpy ndarray -> Matrix")
		A2_*=2.4
		linsys['copy'](A2_,A2)
		linsys['sync'](A2)
		assert array_compare(A2.py,A2_,eps=TEST_EPS)

		PRINT("Copy: col-major numpy ndarray -> Matrix")
		A3_*=2.7
		linsys['copy'](A3_,A2)
		linsys['sync'](A2)
		assert array_compare(A2.py,A3_,eps=TEST_EPS)

		# PRINT("Copy: row-major numpy ndarray->col-major Matrix")
		# A3_ = np.copy(A3.py)
		# A3 *= 0.76
		# copy(A3_,A2)
		# sync(A2)
		# assert array_compare(A2.py,A3_,eps=TEST_EPS)


		# PRINT("Copy: col-major numpy ndarray->col-major Matrix")
		# copy(A3_,A3)
		# sync(A3)
		# assert array_compare(A3.py,A3_,eps=TEST_EPS)

		return True

	except:
		errors.append(format_exc())
		return False

def test_blas_methods(errors, m=4, n=3, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import linsys, Vector, Matrix
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		if m is None: m=4
		if n is None: n=3
		if isinstance(A_in, np.ndarray):
			if len(A_in.shape)==2:
				(m,n) = A_in.shape
				A_in = A_in.astype(backend.lowtypes)
			else:
				A_in=None

		if n < 3 or m < 3: 
			print str("shape provided: ({},{}).\n"
			"using (3,3) as minimum matrix size for tests".format(m,n))
			n=max(n,3)
			m=max(m,3)
			A_in=None

		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid
		PRINTVAR = linsys['print_var'] if VERBOSE_TEST else printvoid
		VEC_ASSERT = lambda *v : var_assert(*v, type=Vector)
		MAT_ASSERT = lambda *A : var_assert(*A, type=Matrix)

		PRINT("BLAS METHODS")


		a=Vector(RAND_ARR(n))
		b=Vector(RAND_ARR(n))
		if isinstance(A_in,np.ndarray):
			A = Matrix(A_in)
		else:
			A=Matrix(RAND_ARR(m,n))
		B=Matrix(RAND_ARR(n,n))
		a_ = np.copy(a.py)
		b_ = np.copy(b.py)
		A_ = np.copy(A.py)
		B_ = np.copy(B.py)
		assert VEC_ASSERT(a,b)
		assert MAT_ASSERT(A,B)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(B.py,B_, eps=TEST_EPS)


		PPRINT("LEVEL 1", '.')

		PRINT("\nVector-vector dot products")
		PRINT("(a,b)")
		res = linsys['dot'](a,b)
		PRINT(res)
		assert abs(res-np.dot(a_,b_)) <= TEST_EPS

		PRINT("\n(a,a)")
		res =linsys['dot'](a,a)
		PRINT(res)
		assert abs(res-np.dot(a_,a_)) <= TEST_EPS

		PRINT("\n2-norm: ||a||_2")
		res = linsys['nrm2'](a)
		PRINT(res)
		assert abs(res-np.linalg.norm(a_,2)) <= TEST_EPS

		PRINT("\n1-norm: ||a||_1")
		res= linsys['asum'](a)
		PRINT(res)
		assert abs(res-np.linalg.norm(a_,1)) <= TEST_EPS


		PRINT("\nBLAS axpy:")
		PRINT("a:")
		PRINTVAR(a)
		PRINT("b:")
		PRINTVAR(b)
		PRINT("b += 3a")
		linsys['axpy'](3,a,b)
		linsys['sync'](a,b)
		b_ += 3*a_
		PRINTVAR(b)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(b.py,b_, eps=TEST_EPS)

		PPRINT("LEVEL 2", '.')

		PRINT("\nBLAS gemv:")
		PRINT("---variable `d`---")
		PRINT("\nAllocation (from {}x1 RAND_ARR array)".format(m))
		d = Vector(RAND_ARR(m))
		assert VEC_ASSERT(d)
		d_ = np.copy(d.py)

		PRINT("d:")
		PRINTVAR(d)
		PRINT("a:")
		PRINTVAR(a)
		assert array_compare(d.py,d_, eps=TEST_EPS)



		PRINT("d := 2.5Aa")
		linsys['gemv']('N',2.5,A,a,0,d)
		linsys['sync'](A,a,d)
		d_ = 2.5*A_.dot(a_)
		PRINTVAR(d)
		PRINT(d.py)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(d.py,d_, eps=TEST_EPS)

		PRINT("d := 3Aa + 2d")
		linsys['gemv']('N',3,A,a,2,d)
		linsys['sync'](A,a,d)
		d_ = 3*A_.dot(a_)+2*d_
		PRINTVAR(d)
		PRINT(d.py)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(a.py,a_, eps=TEST_EPS)
		assert array_compare(d.py,d_, eps=TEST_EPS)

		PRINT("\nBLAS trsv")
		# random lower triangular matrix L
		L_ = RAND_ARR(n,n)
		xrand_ = RAND_ARR(n)
		for i in xrange(n):
			# diagonal entries ~1 (keep condition number reasonable)
			L_[i,i]/=10**np.log(n)
			L_[i,i]+=1.
			# upper triangle = 0
			for j in xrange(n):
				if j>i: L_[i,j]*=0

		xrand = Vector(xrand_)
		L = Matrix(L_)

		PRINT("y = inv(L) * x [py]")
		pysol = np.linalg.solve(L_,xrand_)
		PRINT("y = inv(L) * x [c]")
		backend.dense.blas_trsv(backend.dense_blas_handle, ok_enums.CblasLower, 
			ok_enums.CblasNoTrans, ok_enums.CblasNonUnit,  L.c, xrand.c)
		linsys['sync'](xrand)

		PRINT(pysol)
		PRINTVAR(xrand)
		assert array_compare(xrand.py, pysol, eps=TEST_EPS);


		PPRINT("LEVEL 3",'.')

		PRINT("\nBLAS gemm:")
		PRINT("A:")
		PRINTVAR(A)
		PRINT("B:")
		PRINTVAR(B)

		PRINT("B := 2.13A^TA + 1.05B ")
		linsys['gemm']('T','N',2.13,A,A,1.05,B)
		linsys['sync'](A,B)
		B_ = 2.13*A_.T.dot(A_) + 1.05*B_
		PRINTVAR(B)
		PRINT(B.py)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(B.py,B_, eps=TEST_EPS)

		PRINT("B := A^TA")
		linsys['gemm']('T','N',1,A,A,0,B)
		linsys['sync'](A,B)
		B_ = A_.T.dot(A_)
		PRINTVAR(B)
		PRINT(B.py)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(B.py,B_, eps=TEST_EPS)

		PRINT("\n BLAS syrk")
		PRINT("0.5B += 0.2A^TA")
		B_ *= 0
		B_ += 0.2*A_.T.dot(A_) 
		backend.dense.blas_syrk(backend.dense_blas_handle, ok_enums.CblasLower,
			ok_enums.CblasTrans, 0.2, A.c, 0, B.c)
		linsys['sync'](A,B)

		for i in xrange(n):
			for j in xrange(n):
				if j<=i: continue
				B_[i,j]*=0
				B.py[i,j]*=0

		PRINT(B_)
		PRINTVAR(B)
		assert array_compare(A.py,A_, eps=TEST_EPS)
		assert array_compare(B.py,B_, eps=TEST_EPS)


		if not B.on_gpu:
			PRINT("\n BLAS trsm")
			PRINT("inv(L)*B")
			linsys['sync'](L)
			L_ = np.copy(L.py)
			B_ = np.linalg.inv(L_).dot(B_)
			backend.dense.blas_trsm(backend.dense_blas_handle, ok_enums.CblasLeft, 
				ok_enums.CblasLower, ok_enums.CblasNoTrans, 
				ok_enums.CblasNonUnit,
				1., L.c, B.c)
			linsys['sync'](B)

			PRINT(B_)
			PRINTVAR(B)
			assert array_compare(B.py,B_, eps=TEST_EPS)

		return True

	except:
		errors.append(format_exc())
		return False



def test_linalg_methods(errors, n=10, A_in=None, VERBOSE_TEST=True,
	gpu=False, floatbits=64):

	try: 
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import linsys, Vector, Matrix
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)

		if n is None: n=10
		if isinstance(A_in,np.ndarray):
			if len(A_in.shape)!=2:
				A_in=None
				n=10
			else:
				n=min(A_in.shape[0],A_in.shape[1])
				A_in = A_in.astype(backend.lowtypes.FLOAT_CAST)


		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid
		PRINTVAR = linsys['print_var'] if VERBOSE_TEST else printvoid
		VEC_ASSERT = lambda *v : var_assert(*v, type=Vector)
		MAT_ASSERT = lambda *A : var_assert(*A, type=Matrix)

		PPRINT("LINALG METHODS")

		PRINT("---variable `E`---")
		PRINT("\nAllocation from {} by {} RAND_ARR array".format(n,n))
		if isinstance(A_in,np.ndarray):
			if A_in.shape[0]>=A_in.shape[1]:
				E = Matrix(np.dot(A_in.T,A_in))
			else:
				E = Matrix(np.dot(A_in,A_in.T))
		else:
			F = RAND_ARR(n,n)
			E = Matrix(np.dot(F,F.T))
		assert MAT_ASSERT(E)
		E_ = np.copy(E.py)
		PRINT(E.py)
		assert array_compare(E.py,E_,eps=TEST_EPS)


		PRINT("---variable `x`---")
		PRINT("\nAllocation (from {}x1 RAND_ARR array)".format(n))
		x = Vector(RAND_ARR(n))
		assert VEC_ASSERT(x)
		x_ = np.copy(x.py)
		PRINT(x.py)
		assert array_compare(x.py,x_,eps=TEST_EPS)

		PRINT("\nPython solve: E^-1 x")
		pysol = np.linalg.solve(E_,x_)



		PRINT("\n Python cholesky")
		L_py= np.linalg.cholesky(E_)
		PRINT(L_py)


		PRINT("\nCholesky factorization")
		PRINT("original array:")
		PRINTVAR(E)
			
		linsys['cholesky_factor'](E)
		linsys['sync'](E)
		PRINT("E_LLT := chol(E) (lower triangular)")
		PRINTVAR(E)



		PRINT("L (py) - L (C):")
		L_c = np.zeros_like(L_py)
		for i in xrange(n):
			for j in xrange(n):
				if j > i: continue
				L_c[i,j]=E.py[i,j]


		PRINT(L_py - L_c)
		assert array_compare(L_py,L_c, eps=TEST_EPS)

		PRINT("\nCholesky solve")

		PRINT("before")
		PRINT(x.py)

		PRINT("x := chol_solve(E_LLT,x)")
		linsys['cholesky_solve'](E, x)
		linsys['sync'](x)

		PRINT("after")
		PRINT(x.py)

		PRINT("solve diff (C - py)")
		PRINT(x.py - pysol)
		if backend.lowtypes.FLOAT_CAST == np.float64:
			assert array_compare(x.py, pysol, eps=TEST_EPS)
		else:
			assert array_compare(x.py, pysol, eps=1e-2)

		return True

	except:
		errors.append(format_exc())
		return False

def test_sparse_methods(errors, m=20, n=10, frac_occupied=0.3,
	VERBOSE_TEST=True, gpu=False, floatbits=64):


	try:
		PRINT = println if VERBOSE_TEST else printvoid
		PPRINT = pretty_print if VERBOSE_TEST else printvoid

		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)
		backend.make_linalg_contexts()	

		from optkit.api import linsys, Vector, SparseMatrix
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)


		if m is None: m = 20
		if n is None: n = 10
		if n < 3 or m < 3: 
			print str("shape provided: ({},{}).\n"
			"using (3,3) as minimum matrix size for tests".format(m,n))
			n=max(n,3)
			m=max(m,3)

		if frac_occupied is None: frac_occupied = 0.3
		frac_occupied = max(0., min(1, frac_occupied))
		frac_zero = 1 -  frac_occupied
		
		Arand = RAND_ARR(m, n)
		for i in xrange(m * n):
			Arand[i / n, i % n] *= (np.random.rand() > frac_zero)

		nnz = np.count_nonzero(Arand)


		PPRINT("SPARSE MATRIX METHODS")
		# get sparse handle

		# Allocate all-zero (m, n, nnz) sparse matrix
		PRINT("\nAllocation (int size1={}, size2={}) & printing".format(m,n))
		# matrix
		A = SparseMatrix(m, n, nnz)

		assert var_assert(A)
		# python clone
		A_ = np.copy(A.py)

		PRINT("numpy value:")
		PRINT(A.py)
		PRINT("c value:")
		if VERBOSE_TEST:
			backend.sparse.sp_matrix_print(A.c)

		# set random x, y
		xrand = np.random.rand(n)
		x = Vector(xrand)

		yrand = np.random.rand(m)
		y = Vector(yrand)
		y_empty = Vector(m)		
		x_empty = Vector(n)

		PPRINT("CSR matrix input", '/')

		print Arand 
		
		# Allocate A from CSR matrix
		Acsr = csr_matrix(Arand)
		A = SparseMatrix(Acsr)

		# pdb.set_trace()
	
		# multiply Ax
		PPRINT("y := Ax, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasNoTrans, 1, A.c, x.c, 0, y_empty.c)

		linsys['sync'](y_empty)
		Ax_c = y_empty.py
		Ax_py = Acsr.dot(xrand)
		PRINT("y (Py) - y (C)")
		PRINT(Ax_c - Ax_py)
		assert array_compare(Ax_c, Ax_py, eps=TEST_EPS)

		# multiply A'y
		PPRINT("x := A'y, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasTrans, 1, A.c, y.c, 0, x_empty.c)


		linsys['sync'](x_empty, y)		
		Aty_c = x_empty.py
		Aty_py = Acsr.T.dot(y.py)

		PRINT("x (Py) - x (C)")
		PRINT(Aty_c - Aty_py)
		assert array_compare(Aty_c, Aty_py, eps=TEST_EPS)


		# perform y = 3.1Ax - 2.2y
		PPRINT("y := 3.1Ax - 2.2y, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasNoTrans, 3.1, A.c, x.c, -2.2, y.c)

		linsys['sync'](y)
		Ax_c = y.py		
		Ax_py = 3.1 * Acsr.dot(xrand) - 2.2 * yrand
		PRINT("y (Py) - y (C)")
		PRINT(Ax_c - Ax_py)
		assert array_compare(Ax_c, Ax_py, eps=TEST_EPS)


		PPRINT("CSC matrix input", '/')
		# (update yrand to match y)
		yrand[:] = Ax_py[:] 

		# Allocate A from CSC matrix
		Acsc  = csc_matrix(Arand)
		A = SparseMatrix(Acsc)

		# multiply Ax
		PPRINT("y := Ax, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasNoTrans, 1, A.c, x.c, 0, y_empty.c)

		linsys['sync'](y_empty)
		Ax_c = y_empty.py
		Ax_py = Acsc.dot(xrand)
		PRINT("y (Py) - y (C)")
		PRINT(Ax_c - Ax_py)
		assert array_compare(Ax_c, Ax_py, eps=TEST_EPS)


		# multiply A'y
		PPRINT("x := A'y, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasTrans, 1, A.c, y.c, 0, x_empty.c)

		linsys['sync'](x_empty, y)		
		Aty_c = x_empty.py
		Aty_py = Acsc.T.dot(y.py)

		PRINT("x (Py) - x (C)")
		PRINT(Aty_c - Aty_py)
		assert array_compare(Aty_c, Aty_py, eps=TEST_EPS)


		# perform y = 3.1Ax - 2.2y
		PPRINT("y := 3.1Ax - 2.2y, python vs C", '.')
		backend.sparse.sp_blas_gemv(backend.sparse_handle, 
			ok_enums.CblasNoTrans, 3.1, A.c, x.c, -2.2, y.c)

		linsys['sync'](y)
		Ax_c = y.py		
		Ax_py = 3.1 * Acsc.dot(xrand) - 2.2 * yrand
		PRINT("y (Py) - y (C)")
		PRINT(Ax_c - Ax_py)
		assert array_compare(Ax_c, Ax_py, eps=TEST_EPS)


		return True
	except:
		errors.append(format_exc())
		return False


def test_linsys(errors, *args,**kwargs):
	print "LINSYS METHODS TESTING\n\n\n\n"
	tests = 0
	args = list(args)
	verbose = '--verbose' in args
	(m,n) = kwargs['shape'] if 'shape' in kwargs else (None,None)
	A = np.load(kwargs['file']) if 'file' in kwargs else None
	floatbits = 32 if 'float' in args else 64
	sparse_occupancy = kwargs['occupancy'] if 'occupancy' in kwargs else 0.3

	success = True

	if '--allsub' in args:
		args+=['--lowvec','--lowmat','--lowsparse',
		'--vec','--mat','--blas','--linalg','--sparse']

	if '--lowvec' in args:
		tests += 1
		success &= test_lowlevelvectorcalls(errors, VERBOSE_TEST=verbose,
			gpu='gpu' in args, floatbits=floatbits)
	if '--lowmat' in args:
		tests += 1
		success &= test_lowlevelmatrixcalls(errors, VERBOSE_TEST=verbose,
			gpu='gpu' in args, floatbits=floatbits)
	if '--lowsparse' in args:
		tests += 1
		success &= test_lowlevelsparsecalls(errors, VERBOSE_TEST=verbose,
			gpu='gpu' in args, floatbits=floatbits)
	if '--vec' in args:
		tests += 1
		success &= test_vector_methods(errors, n=n, VERBOSE_TEST=verbose,
			gpu='gpu' in args, floatbits=floatbits)
	if '--mat' in args:
		tests += 1
		success &= test_matrix_methods(errors, m=m, n=n, 
			VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)
	if '--blas' in args:
		tests += 1
		success &= test_blas_methods(errors, m=m, n=n, A_in=A, 
			VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)
	if '--linalg' in args:
		tests += 1
		success &= test_linalg_methods(errors, n=n, A_in=A, 
			VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)
	if '--sparse' in args:
		tests += 1
		success &= test_sparse_methods(errors, m=m, n=n, 
			frac_occupied=sparse_occupancy,
			VERBOSE_TEST=verbose, gpu='gpu' in args, floatbits=floatbits)

	terminal_char = '' if tests == 1 else 's'
	print "{} sub-test{} completed".format(tests, terminal_char)
	if tests == 0:
		print str("no linear systems tests specified."
			"\nuse optional arguments:\n"
			"--lowvec,\n--lowmat,\n--lowsparse,\n"
			"--vec,\n--mat,\n--blas,\n--linalg,\n"
			"--sparse,\n or\n--allsub\n to specify tests.")

	if success:
		print "...passed"
	else:
		print "...failed"
	return success