from optkit.types import ok_enums as enums
from optkit.types.highlevel import Range
from optkit.utils import istypedtuple
from numpy import ndarray 
import sys

class LinsysCoreKernels(object):
	
	def __init__(self, backend, vector_type, matrix_type, sparse_matrix_type):
		if not backend.linalg_contexts_exist:
			backend.make_linalg_contexts()

		self.on_gpu = backend.device == 'gpu'
		dense_blas_handle = backend.dense_blas_handle
		denselib = backend.dense

		self.float = backend.lowtypes.FLOAT_CAST
		ndarray_pointer = backend.lowtypes.ndarray_pointer
 		Vector = vector_type
		Matrix = matrix_type
		SparseMatrix = sparse_matrix_type

		dimcheck = backend.dimcheck
		typecheck = backend.typecheck
		devicecheck = backend.devicecheck

		def device_compare(*args):
			for arg in args:
				if arg.on_gpu != self.on_gpu:
					raise ValueError("kernel call on GPU ={}\n"
						"all inputs on GPU = {}".format(self.on_gpu, arg.on_gpu))
		self.device_compare = device_compare

		def set_all(a, x):
			if not isinstance(x, (Vector)):
				raise TypeError("input 'x' must be an optkit.Vector.\n"
								"Provided:{}".format(type(x)))
			elif not isinstance(a, (int, float, self.float)):
				raise TypeError("input 'a' must be a (real) scalar\n"
								"Provided:{}".format(type(x)))
			else: 
				if devicecheck: device_compare(x)
				denselib.vector_set_all(x.c,a)
		self.set_all = set_all

		def copy(orig, dest):
			if isinstance(orig, Vector) and isinstance(dest, Vector):
				if	devicecheck: device_compare(orig, dest)
				denselib.vector_memcpy_vv(dest.c,orig.c)
			elif isinstance(orig, Matrix) and isinstance(dest, Matrix):
				if	devicecheck: device_compare(orig, dest)
				denselib.matrix_memcpy_mm(dest.c,orig.c)
			elif isinstance(orig, ndarray) and isinstance(dest, Vector):
				orig = self.float(orig)
				if dimcheck:
					if orig.size != dest.size: raise ValueError(
					"incompatible array shapes for copy")
				if	devicecheck: device_compare(orig, dest)

				denselib.vector_memcpy_va(dest.c, ndarray_pointer(
					orig), orig.strides[0]/orig.itemsize)
			elif isinstance(orig, ndarray) and isinstance(dest, Matrix):
				orig = self.float(orig)
				if dimcheck:
					if orig.shape != dest.shape: raise ValueError(
					"incompatible array shapes for copy")

				if	devicecheck: device_compare(orig, dest)

				order = enums.CblasRowMajor if orig.flags.c_contiguous else \
						enums.CblasColMajor
				denselib.matrix_memcpy_ma(dest.c, ndarray_pointer(
					orig), order)
			else:
				raise TypeError("optkit.kernels.linsys.copy(dest, orig) defined "
					  "only when arguments are type:\n\t"
					  "(optkit.Vector, optkit.Vector)\n\t"	
					  "(optkit.Matrix, optkit.Matrix)\n\t"
					  "(optkit.SparseMatrix, optkit.SparseMatrix)\n\t"	
					  "(numpy.ndarray, optkit.Vector)\n\t"	
					  "(numpy.ndarray, optkit.Matrix)\n\t"
					  "(scipy.sparse.csr_matrix, optkit.SparseMatrix)\n\t"
					  "(scipy.sparse.csc_matrix, optkit.SparseMatrix).")	
		self.copy = copy 

		def view(x, *range_, **viewtype):
			input_err = str("optkit.kernels.linsys.view: "
				"invalid view specification.\n"
				"Valid argument & keyword argument combinations:\n"
				"(`optkit.Vector`, `tuple(int,int)`)\n"
				"(`optkit.Matrix`, diag=1)\n"
				"(`optkit.Matrix`, `int`, [no keyword args, views row])\n"		
				"(`optkit.Matrix`, `int`, row=1)\n"
				"(`optkit.Matrix`, `int`, col=1)\n"
				"(`optkit.Matrix`, `tuple(int,int)`, `tuple(int,int)`)\n"
				"Provided: x:{}\n args:{}\n kwargs:{}".format(
					type(x),
					[type(r) for r in range_],
					["{}={}".format(k,viewtype[k]) for k in viewtype.keys()],
					))

			if devicecheck: device_compare(x)

			if not isinstance(x, (Vector, Matrix)):
				raise TypeError("optkit.kernels.linsys.view(x) only defined "
					  "for argument of type optkit.Vector or "
					  "optkit.Matrix.")

			elif isinstance(x, Vector) and \
				 len(range_) == 1 and \
				 istypedtuple(range_[0],2,int):

				rng = Range(x.size, *range_[0])
				pyview = x.py[rng.idx1:rng.idx2]
				cview = backend.lowtypes.vector(0, 0, None)
				denselib.vector_subvector(cview, x.c, rng.idx1, rng.elements)
				return Vector(pyview, cview, is_view=1)
			elif isinstance(x, Matrix) and \
				 len(range_) == 2 and  \
				 istypedtuple(range_,2,tuple):

				if not istypedtuple(range_[0],2,int) and \
					   istypedtuple(range_[1],2,int):

					raise TypeError(input_err)

				rng1 = Range(x.size1, *range_[0])
				rng2 = Range(x.size2, *range_[1])		
				pyview = x.py[rng1.idx1:rng1.idx2,rng2.idx1:rng2.idx2]
				cview = backend.lowtypes.matrix(0, 0, 0, None, x.c.order)
				denselib.matrix_submatrix(cview,x.c, 
										 rng1.idx1, rng2.idx1, 
										 rng1.elements, rng2.elements)		
				return Matrix(pyview, cview, is_view=1)
			elif isinstance(x, Matrix) and len(range_) == 1:
				idx = range_[0]
				cview = backend.lowtypes.vector(0, 0, None)
				if 'col' in viewtype:
					col = Range(x.size2, idx).idx1
					denselib.matrix_column(cview, x.c, col)
					pyview = x.py[:,col]
				else:
					row = Range(x.size1,idx).idx1
					denselib.matrix_row(cview, x.c, row)
					pyview = x.py[row,:]
					if not 'row' in viewtype:
						Warning("keyword argument `row=1`, `col=1` or `diag=1` "
						  "not provided, assuming row view")
				return Vector(pyview,cview, is_view=1)			
			elif 'diag' in viewtype:
				cview = backend.lowtypes.vector(0, 0, None)
				denselib.matrix_diagonal(cview, x.c)
				pyview = x.py.diagonal().copy()
				return Vector(pyview, cview, sync_required=1, is_view=1)
			else: 
				raise TypeError(input_err)

		self.view = view

		"""
		keyword args: python_to_C (default=False), sets copy direction
		"""
		def sync(*vars, **py2c):
			python_to_C = "python_to_C" in py2c

			for x in vars:
				if not isinstance(x, (Vector, Matrix)):
					raise TypeError("optkit.kernels.linsys.sync undefined for "
						  "types other than:\n optkit.Vector "
						  "\n optkit.Matrix.")	
				else:
					if not x.sync_required: return
					if	devicecheck: device_compare(x)

					if isinstance(x, Vector):
						if python_to_C:
							denselib.vector_memcpy_va(x.c, ndarray_pointer(x.py),
								x.py.strides[0]/x.py.itemsize)
						else:
							denselib.vector_memcpy_av(ndarray_pointer(x.py), x.c, 
								x.py.strides[0]/x.py.itemsize)
					else:
						order = enums.CblasRowMajor if x.py.flags.c_contiguous \
							else enums.CblasColMajor

						if python_to_C:
							denselib.matrix_memcpy_ma(x.c, ndarray_pointer(x.py), order)
						else:
							denselib.matrix_memcpy_am(ndarray_pointer(x.py), x.c, order)

		self.sync = sync

		def print_var(x):
			if not isinstance(x, (Vector, Matrix)):
				raise TypeError("optkit.kernels.linsys.print_var undefined for "
					   "types other than: \n optkit.Vector"
						"\n optkit.Matrix.")
			if	devicecheck: device_compare(x)


			if isinstance(x, Vector):
				denselib.vector_print(x.c)
			else:
				denselib.matrix_print(x.c)

		self.print_var = print_var

		def add(const_x, y):
			if isinstance(const_x, Vector) and isinstance(y, Vector):
				if	devicecheck: device_compare(const_x, y)
				if dimcheck and y.size != const_x.size: 
					raise ValueError("optkit.kernels.linsys.add---"
						   "incompatible Vector dimensions\n"
						   "const_x: {}, y: {}".format(const_x.size, y.size))

				denselib.vector_add(y.c, const_x.c);
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Vector):
				if	devicecheck: device_compare(y)
				denselib.vector_add_constant(y.c, const_x)
			else:
				raise TypeError("optkit.kernels.linsys.add(x,y) defined for : \n"
					  "\t(optkit.Vector, optkit.Vector) \n"
					  "\t(int/float, optkit.Vector).")

		self.add = add

		def sub(const_x, y):
			if isinstance(const_x, Vector) and isinstance(y, Vector):
				if	devicecheck: device_compare(const_x, y)
				if dimcheck and y.size != const_x.size: 
					raise ValueError("Error: optkit.kernels.linsys.sub---"
						   "incompatible Vector dimensions\n"
						   "const_x: {}, y: {}".format(const_x.size, y.size))

				denselib.vector_sub(y.c, const_x.c);
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Vector):
				if	devicecheck: device_compare(y)
				denselib.vector_add_constant(y.c, -const_x)
			else:
				raise TypeError("optkit.kernels.linsys.sub(x,y) defined for : \n"
					  "\t(optkit.Vector, optkit.Vector) \n"
					  "\t(int/float, optkit.Vector).")

		self.sub = sub

		def mul(const_x, y):
			if isinstance(const_x, Vector) and isinstance(y, Vector):
				if	devicecheck: device_compare(const_x, y)
				if dimcheck and y.size != const_x.size: 
					raise ValueError("Error: optkit.kernels.linsys.mul---"
						   "incompatible Vector dimensions\n"
						   "const_x: {}, y: {}".format(const_x.size, y.size))
				denselib.vector_mul(y.c, const_x.c);
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Vector):
				if	devicecheck: device_compare(y)
				denselib.vector_scale(y.c, const_x);
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Matrix):
				if	devicecheck: device_compare(y)
				denselib.matrix_scale(y.c, const_x);		
			else:
				raise TypeError("optkit.kernels.linsys.mul(x,y) defined for : \n"
					  "\t(optkit.Vector, optkit.Vector) \n"
					  "\t(int/float, optkit.Vector) \n"
					  "\t(int/float, optkit.Matrix)\n"
					  "Provided:{}{}".format(type(const_x),type(y)))

		self.mul = mul

		def div(const_x, y):
			if isinstance(const_x, Vector) and isinstance(y, Vector):
				if	devicecheck: device_compare(const_x, y)
				if dimcheck and y.size != const_x.size: 
					raise ValueError("Error: optkit.kernels.linsys.div---"
						   "incompatible Vector dimensions\n"
						   "const_x: {}, y: {}".format(const_x.size, y.size))
				else:
					denselib.vector_div(y.c, const_x.c);
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Vector):
				if	devicecheck: device_compare(y)
				denselib.vector_scale(y.c, 1./const_x);		
			elif isinstance(const_x, (int, float, self.float)) and isinstance(y, Matrix):
				if	devicecheck: device_compare(y)
				denselib.matrix_scale(y.c, 1./const_x);		
			else:
				raise TypeError("optkit.kernels.linsys.div(x,y) defined for : \n"
					  "\t(optkit.Vector, optkit.Vector) \n"
					  "\t(int/float, optkit.Vector) \n"
					  "\t(int/float, optkit.Matrix).")

		self.div = div

		def elemwise_inverse(v):
			pass

		self.elemwise_inverse = elemwise_inverse

		def elemwise_sqrt(v):	
			pass

		self.elemwise_sqrt = elemwise_sqrt

		def elemwise_inverse_sqrt(v):	
			pass

		self.elemwise_inverse_sqrt = elemwise_inverse_sqrt

		def dot(x, y):
			if typecheck and not \
				   (isinstance(x, Vector) and 
					isinstance(y, Vector)):
				raise TypeError("optkit.kernels.linsys.dot(x,y) defined for : \n"
					  "\t(optkit.Vector, optkit.Vector).")
			
			if dimcheck and y.size != x.size: 
				raise ValueError("optkit.kernels.linsys.dot---"
					   "incompatible Vector dimensions\n"
					   "x: {}, y: {}".format(x.size, y.size))
		 	
			if	devicecheck: device_compare(x, y)

		 	return denselib.blas_dot(dense_blas_handle, x.c,y.c)

		self.dot = dot

		def asum(x):
			if typecheck and not isinstance(x, Vector):
				raise TypeError("optkit.kernels.linsys.asum(x) defined for "
					"optkit.Vector.")

			if	devicecheck: device_compare(x)

			return denselib.blas_asum(dense_blas_handle, x.c)

		self.asum = asum

		def nrm2(x):
			if typecheck and not isinstance(x, Vector):
				raise TypeError("optkit.kernels.linsys.nrm2(x) defined for "
					"optkit.Vector.")

			if	devicecheck: device_compare(x)

			return denselib.blas_nrm2(dense_blas_handle, x.c)

		self.nrm2 = nrm2

		def axpy(alpha, const_x, y):
			if typecheck:
				valid = isinstance(alpha, (int, float, self.float))
				valid &= isinstance(const_x, Vector)
				valid &= isinstance(y, Vector)
				if not valid:
					raise TypeError ("optkit.kernels.linsys.axpy(alpha, x, y) " 
						"defined for: \n\t(int/float, optkit.Vector, optkit.Vector).")
			if dimcheck and const_x.size != y.size:
					raise ValueError("optkit.kernels.linsys.axpy---"
						   "incompatible dimensions for y+=alpha x\n"
						   "x: {}, y: {}".format(const_x.size, y.size))

			if	devicecheck: device_compare(const_x, y)


			denselib.blas_axpy(dense_blas_handle, alpha, const_x.c, y.c)			

		self.axpy = axpy

		def gemv(tA, alpha, A, x, beta, y):
			if typecheck:
				valid = isinstance(alpha, (int, float, self.float))
				valid &= isinstance(A, Matrix)   
				valid &= isinstance(x, Vector) 
				valid &= isinstance(beta, (int, float, self.float))
				valid &= isinstance(y, Vector)
				if not valid:
					raise TypeError(
					"optkit.kernels.linsys.gemv(ta,alpha, A, x, beta, y) " 
					"defined for : \n\t(str, int/float, optkit.Matrix, "
					"optkit.Vector, int/float, optkit.Vector).\nProvided:"
					"\n\t({},{},{},{},{},{})".format(type(tA),type(alpha),
						type(A),type(x),type(beta),type(y)))

			if dimcheck:
				input_dim = A.size1 if tA=='T' else A.size2
				output_dim = A.size2 if tA=='T' else A.size1
				tsym = "^T" if tA=='T' else ""
				if  tA == 'T':
					dim_in = A.size1
					dim_out = A.size2
					tsym = "^T"
				else:
					dim_in = A.size2
					dim_out = A.size1
					tsym = ""
				if (x.size!= dim_in or y.size != dim_out): 
					raise ValueError("optkit.kernels.linsys.gemv---"
					   "incompatible dimensions for y=A{} * x\n"
					   "A: {},{}\n x: {}, y: {}".format(tsym,
					   	A.size1, A.size2, x.size, y.size))

			if	devicecheck: device_compare(A, x, y)


			At = enums.CblasTrans if tA =='T' else enums.CblasNoTrans			
			denselib.blas_gemv(dense_blas_handle, At, alpha, A.c, x.c, beta, y.c)

		self.gemv = gemv

		def gemm(tA, tB, alpha, A, B, beta, C):
			if typecheck:
				valid=isinstance(alpha, (int, float, self.float))
				valid &= isinstance(A, Matrix)  
				valid &= isinstance(B, Matrix) 
				valid &= isinstance(beta, (int, float, self.float)) 
				valid &= isinstance(C, Matrix)
				if not valid: 
					raise TypeError(
					"optkit.kernels.linsys.gemm(tA,tB,alpha, A, B, beta, C) "
					"defined for:\n\t(str,str,int/float, optkit.Matrix, " 
					"optkit.Matrix, int/float, optkit.Matrix)\nProvided:" 
					"\n\t({},{},{},{},{},{},{}).".format(
						type(tA),type(tB),type(alpha),type(A),
						type(B),type(beta),type(C)))
			if dimcheck:
				outer_dim_L = A.size2 if tA=='T' else A.size1
				inner_dim_L = A.size1 if tA=='T' else A.size2
				inner_dim_R = B.size2 if tB=='T' else B.size1
				outer_dim_R = B.size1 if tB=='T' else B.size2
				tsymA = "^T" if tA=='T' else ""
				tsymB = "^T" if tB=='T' else ""


				if (C.size1 != outer_dim_L or \
							 inner_dim_L != inner_dim_R or \
							 C.size2 != outer_dim_R): 
					raise ValueError("Error: optkit.kernels.linsys.gemm---"
					   "incompatible dimensions for C=A{} * B{}\n"
					   "A: {}x{}\nB: {}x{}\nC: {}x{}".format(
					   	tsymA, tsymB, A.size1, A.size2, B.size1, 
					   	B.size2, C.size1, C.size2))


			if	devicecheck: device_compare(A, B, C)

				
			At = enums.CblasTrans if tA =='T' else enums.CblasNoTrans
			Bt = enums.CblasTrans if tB =='T' else enums.CblasNoTrans

			denselib.blas_gemm(dense_blas_handle, At, Bt, alpha, A.c, B.c, beta, C.c)

		self.gemm = gemm

		def cholesky_factor(A):
			if typecheck and not isinstance(A, Matrix):
				raise TypeError("optkit.kernels.linsys.cholesky_factor(A) defined"
				      "only when argument is of"
					  "type opkit.Matrix.")
			if dimcheck and A.size1 != A.size2:
				raise ValueError("optkit.kernels.linsys.cholesky_factor(A)"
					   "only defined for square matrices A"
					   "A: {}x{}".format(A.size1, A.size2))

			if	devicecheck: device_compare(A)

			denselib.linalg_cholesky_decomp(dense_blas_handle, A.c)


		self. cholesky_factor = cholesky_factor

		def cholesky_solve(L, x):
			if typecheck:
				if not isinstance(L, Matrix):
					raise TypeError("optkit.kernels.linsys.cholesky_solve(L, x) defined"
						  "only when first argument is of"
						  "type opkit.Matrix.")
				elif not isinstance(x, Vector):
					raise TypeError("optkit.kernels.linsys.cholesky_solve(L, x) defined"
						  "only when second argument is of"
						  "type opkit.Vector.")

			if dimcheck and (x.size != L.size2 or x.size != L.size2): 
				raise ValueError("Error: optkit.kernels.linsys.cholesky_solve---"
					   "incompatible dimensions for x:=inv(L) * x\n"
					   "L: {}x{}\nx: {}".format(L.size1, L.size2, x.size))

			if	devicecheck: device_compare(L, x)

			denselib.linalg_cholesky_svx(dense_blas_handle, L.c,x.c)

		self.cholesky_solve = cholesky_solve
