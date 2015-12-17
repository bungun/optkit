from optkit.types import ok_enums as enums
from optkit.types.highlevel import Range
from optkit.utils import istypedtuple
from ctypes import c_void_p, byref
from numpy import ndarray 
import sys

class LinsysCoreKernels(object):
	
	def __init__(self, backend, vector_type, matrix_type):
		self.dense_blas_handle = backend.dense_blas_handle
		self.denselib = backend.dense
		self.dimcheck = backend.dimcheck_default
		self.typecheck = backend.typecheck_default
		self.ndarray_pointer = backend.lowtypes.ndarray_pointer
		self.make_cvector = backend.make_cvector
		self.make_cmatrix = backend.make_cmatrix
		self.Vector = vector_type
		self.Matrix = matrix_type


		self.CHK_MSG = str("\nMake sure to not mix backends: device "
			"(CPU vs. GPU) and floating pointer precision (32- vs 64-bit) "
			"must match.\n Current settings: {}-bit precision, "
			"{}.".format(backend.precision, backend.device))

	def set_all(self, a, x):
		if not isinstance(x, (self.Vector)):
			raise TypeError("input 'x' must be an optkit.Vector.\n"
							"Provided:{}.{}".format(type(x), self.CHK_MSG))
		elif not isinstance(a, (int,float)):
			raise TypeError("input 'a' must be a (real) scalar\n"
							"Provided:{}".format(type(x)))
		else: self.denselib.vector_set_all(x.c,a)

	def copy(self, orig, dest):
		if isinstance(orig, self.Vector) and isinstance(dest, self.Vector):
			self.denselib.vector_memcpy_vv(dest.c,orig.c)
		elif isinstance(orig, self.Matrix) and isinstance(dest, self.Matrix):
			self.denselib.matrix_memcpy_mm(dest.c,orig.c)
		elif isinstance(orig, ndarray) and isinstance(dest, self.Vector):
			if orig.size != dest.size: raise ValueError(
				"incompatible array shapes for copy")
			self.denselib.vector_memcpy_va(dest.c, self.ndarray_pointer(orig),
				orig.strides[0]/orig.itemsize)
		elif isinstance(orig, ndarray) and isinstance(dest, self.Matrix):
			if orig.shape != dest.shape: raise ValueError(
				"incompatible array shapes for copy")
			order = enums.CblasRowMajor if orig.flags.c_contiguous else \
					enums.CblasColMajor
			self.denselib.matrix_memcpy_ma(dest.c, self.ndarray_pointer(orig), order)
		else:
			raise TypeError("optkit.kernels.linsys.copy(dest, orig) defined "
				  "only when arguments are type:\n\t"
				  "(optkit.Vector,optkit.Vector\n\t"	
				  "(optkit.Matrix,optkit.Matrix\n\t"	
				  "(numpy.ndarray,optkit.Vector\n\t"	
				  "(numpy.ndarray,optkit.Matrix.\n{}".format(self.CHK_MSG))	

	def view(self, x, *range_, **viewtype):
		input_err = str("optkit.kernels.linsys.view: "
			"invalid view specification.\n"
			"Valid argument & keyword argument combinations:\n"
			"(`optkit.Vector`, `tuple(int,int)`)\n"
			"(`optkit.Matrix`, diag=1)\n"
			"(`optkit.Matrix`, `int`, [no keyword args, views row])\n"		
			"(`optkit.Matrix`, `int`, row=1)\n"
			"(`optkit.Matrix`, `int`, col=1)\n"
			"(`optkit.Matrix`, `tuple(int,int)`, `tuple(int,int)`)\n"
			"Provided: x:{}\n args:{}\n kwargs:{}.{}".format(
				type(x),
				[type(r) for r in range_],
				["{}={}".format(k,viewtype[k]) for k in viewtype.keys()],
				self.CHK_MSG))


		if not isinstance(x, (self.Vector, self.Matrix)):
			raise TypeError("optkit.kernels.linsys.view(x) only defined "
				  "for argument of type optkit.Vector or "
				  "optkit.Matrix.")

		elif isinstance(x, self.Vector) and \
			 len(range_) == 1 and \
			 istypedtuple(range_[0],2,int):

			rng = Range(x.size, *range_[0])
			pyview = x.py[rng.idx1:rng.idx2]
			cview = self.make_cvector()
			self.denselib.vector_subvector(cview, x.c, rng.idx1, rng.elements)
			return self.Vector(pyview, cview, is_view=1)
		elif isinstance(x, self.Matrix) and \
			 len(range_) == 2 and  \
			 istypedtuple(range_,2,tuple):

			if not istypedtuple(range_[0],2,int) and \
				   istypedtuple(range_[1],2,int):

				raise TypeError(input_err)

			rng1 = Range(x.size1, *range_[0])
			rng2 = Range(x.size2, *range_[1])		
			pyview = x.py[rng1.idx1:rng1.idx2,rng2.idx1:rng2.idx2]
			cview = self.make_cmatrix()
			self.denselib.matrix_submatrix(cview,x.c, 
									 rng1.idx1, rng2.idx1, 
									 rng1.elements, rng2.elements)		
			return self.Matrix(pyview, cview, is_view=1)
		elif isinstance(x, self.Matrix) and len(range_) == 1:
			idx = range_[0]
			cview = self.make_cvector()
			if 'col' in viewtype:
				col = Range(x.size2, idx).idx1
				self.denselib.matrix_column(cview, x.c, col)
				pyview = x.py[:,col]
			else:
				row = Range(x.size1,idx).idx1
				self.denselib.matrix_row(cview, x.c, row)
				pyview = x.py[row,:]
				if not 'row' in viewtype:
					Warning("keyword argument `row=1`, `col=1` or `diag=1` "
					  "not provided, assuming row view")
			return self.Vector(pyview,cview, is_view=1)			
		elif 'diag' in viewtype:
			cview = self.make_cvector()
			self.denselib.matrix_diagonal(cview, x.c)
			pyview = x.py.diagonal().copy()
			return self.Vector(pyview, cview, sync_required=1, is_view=1)
		else: 
			raise TypeError(input_err)


	"""
	keyword args: python_to_C (default=False), sets copy direction
	"""
	def sync(self, *vars, **py2c):
		python_to_C = "python_to_C" in py2c

		for x in vars:
			if not isinstance(x, (self.Vector, self.Matrix)):
				raise TypeError("optkit.kernels.linsys.sync undefined for "
					  "types other than:\n optkit.Vector "
					  "\n optkit.Matrix. {}".format(self.CHK_MSG))	
			else:
				if not x.sync_required: return
				if isinstance(x, self.Vector):
					if python_to_C:
						self.denselib.vector_memcpy_va(x.c, self.ndarray_pointer(x.py),
							x.py.strides[0]/x.py.itemsize)
					else:
						self.denselib.vector_memcpy_av(self.ndarray_pointer(x.py), x.c, 
							x.py.strides[0]/x.py.itemsize)
				else:
					order = enums.CblasRowMajor if x.py.flags.c_contiguous \
						else enums.CblasColMajor

					if python_to_C:
						self.denselib.matrix_memcpy_ma(x.c, self.ndarray_pointer(x.py), order)
					else:
						self.denselib.matrix_memcpy_am(self.ndarray_pointer(x.py), x.c, order)



	def print_var(self, x):
		if not isinstance(x, (self.Vector, self.Matrix)):
			raise TypeError("optkit.kernels.linsys.print_var undefined for "
				   "types other than: \n optkit.Vector"
					"\n optkit.Matrix. {}".format(self.CHK_MSG))
		else:
			if python:
				if x.sync_required: sync(x)
				print x.py
			elif isinstance(x, self.Vector):
				self.denselib.vector_print(x.c)
			else:
				self.denselib.matrix_print(x.c)


	def add(self, const_x,y):
		if isinstance(const_x, self.Vector) and isinstance(y, self.Vector):
			if y.size != const_x.size: 
				raise ValueError("optkit.kernels.linsys.add---"
					   "incompatible Vector dimensions\n"
					   "const_x: {}, y: {}".format(const_x.size, y.size))
			else:
				self.denselib.vector_add(y.c, const_x.c);
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Vector):
			self.denselib.vector_add_constant(y.c, const_x)
		else:
			raise TypeError("optkit.kernels.linsys.add(x,y) defined for : \n"
				  "\t(optkit.Vector, optkit.Vector) \n"
				  "\t(int/float, optkit.Vector). {}".format(self.CHK_MSG))


	def sub(self, const_x,y):
		if isinstance(const_x, self.Vector) and isinstance(y, self.Vector):
			if y.size != const_x.size: 
				raise ValueError("Error: optkit.kernels.linsys.sub---"
					   "incompatible Vector dimensions\n"
					   "const_x: {}, y: {}".format(const_x.size, y.size))
			else:
				self.denselib.vector_sub(y.c, const_x.c);
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Vector):
			self.denselib.vector_add_constant(y.c, -const_x)
		else:
			raise TypeError("optkit.kernels.linsys.sub(x,y) defined for : \n"
				  "\t(optkit.Vector, optkit.Vector) \n"
				  "\t(int/float, optkit.Vector). {}".format(self.CHK_MSG))


	def mul(self, const_x,y):
		if isinstance(const_x, self.Vector) and isinstance(y, self.Vector):
			if y.size != const_x.size: 
				raise ValueError("Error: optkit.kernels.linsys.mul---"
					   "incompatible Vector dimensions\n"
					   "const_x: {}, y: {}".format(const_x.size, y.size))
			else:
				self.denselib.vector_mul(y.c, const_x.c);
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Vector):
			self.denselib.vector_scale(y.c, const_x);
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Matrix):
			self.denselib.matrix_scale(y.c, const_x);		
		else:
			raise TypeError("optkit.kernels.linsys.mul(x,y) defined for : \n"
				  "\t(optkit.Vector, optkit.Vector) \n"
				  "\t(int/float, optkit.Vector) \n"
				  "\t(int/float, optkit.Matrix)\n"
				  "Provided:{}{}{}".format(type(const_x),type(y),self.CHK_MSG))


	def div(self, const_x,y):
		if isinstance(const_x, self.Vector) and isinstance(y, self.Vector):
			if y.size != const_x.size: 
				raise ValueError("Error: optkit.kernels.linsys.div---"
					   "incompatible Vector dimensions\n"
					   "const_x: {}, y: {}".format(const_x.size, y.size))
			else:
				self.denselib.vector_div(y.c, const_x.c);
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Vector):
			self.denselib.vector_scale(y.c, 1./const_x);		
		elif isinstance(const_x, (int,float)) and isinstance(y, self.Matrix):
			self.denselib.matrix_scale(y.c, 1./const_x);		
		else:
			raise TypeError("optkit.kernels.linsys.div(x,y) defined for : \n"
				  "\t(optkit.Vector, optkit.Vector) \n"
				  "\t(int/float, optkit.Vector) \n"
				  "\t(int/float, optkit.Matrix). {}".format(self.CHK_MSG))

	def elemwise_inverse(self, v):
		pass

	def elemwise_sqrt(self, v):	
		pass

	def elemwise_inverse_sqrt(self, v):	
		pass

	def dot(self, x,y, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck and not \
			   (isinstance(x, self.Vector) and 
				isinstance(y, self.Vector)):
			raise TypeError("optkit.kernels.linsys.dot(x,y) defined for : \n"
				  "\t(optkit.Vector, optkit.Vector). {}".format(self.CHK_MSG))
		
		if dimcheck and y.size != x.size: 
			raise ValueError("optkit.kernels.linsys.dot---"
				   "incompatible Vector dimensions\n"
				   "x: {}, y: {}".format(x.size, y.size))
	 	return self.denselib.blas_dot(self.dense_blas_handle, x.c,y.c)

	def asum(self, x, typecheck=True):
		if typecheck is None: typecheck = self.typecheck
		if typecheck and not isinstance(x, self.Vector):
			raise TypeError("optkit.kernels.linsys.asum(x) defined for "
				"optkit.Vector. {}".format(self.CHK_MSG))
		else:
			return self.denselib.blas_asum(self.dense_blas_handle, x.c)

	def nrm2(self, x, typecheck=True):
		if typecheck is None: typecheck = self.typecheck
		if typecheck and not isinstance(x, self.Vector):
			raise TypeError("optkit.kernels.linsys.nrm2(x) defined for "
				"optkit.Vector. {}".format(self.CHK_MSG))
		else:
			return self.denselib.blas_nrm2(self.dense_blas_handle, x.c)

	def axpy(self, alpha, const_x, y, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck:
			valid = isinstance(alpha, (int,float))
			valid &= isinstance(const_x, self.Vector)
			valid &= isinstance(y, self.Vector)
			if not valid:
				raise TypeError ("optkit.kernels.linsys.axpy(alpha, x, y) " 
					"defined for: \n\t(int/float, optkit.Vector, optkit.Vector). {}".format(
						self.CHK_MSG))
		if dimcheck and const_x.size != y.size:
				raise ValueError("optkit.kernels.linsys.axpy---"
					   "incompatible dimensions for y+=alpha x\n"
					   "x: {}, y: {}".format(const_x.size, y.size))
		self.denselib.blas_axpy(self.dense_blas_handle, alpha, const_x.c, y.c)			

	def gemv(self, tA, alpha, A, x, beta, y, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck:
			valid = isinstance(alpha, (int,float))
			valid &= isinstance(A, self.Matrix)   
			valid &= isinstance(x, self.Vector) 
			valid &= isinstance(beta, (int,float))
			valid &= isinstance(y, self.Vector)
			if not valid:
				raise TypeError(
				"optkit.kernels.linsys.gemv(ta,alpha, A, x, beta, y) " 
				"defined for : \n\t(str, int/float, optkit.Matrix, "
				"optkit.Vector, int/float, optkit.Vector).\nProvided:"
				"\n\t({},{},{},{},{},{}). \n{}".format(type(tA),type(alpha),
					type(A),type(x),type(beta),type(y), self.CHK_MSG))

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

		At = enums.CblasTrans if tA =='T' else enums.CblasNoTrans			
		self.denselib.blas_gemv(self.dense_blas_handle, At, alpha, A.c, x.c, beta, y.c)


	def gemm(self, tA, tB, alpha, A, B, beta, C, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck:
			valid=isinstance(alpha, (int,float))
			valid &= isinstance(A, self.Matrix)  
			valid &= isinstance(B, self.Matrix) 
			valid &= isinstance(beta, (int,float)) 
			valid &= isinstance(C, self.Matrix)
			if not valid: 
				raise TypeError(
				"optkit.kernels.linsys.gemm(tA,tB,alpha, A, B, beta, C) "
				"defined for:\n\t(str,str,int/float, optkit.Matrix, " 
				"optkit.Matrix, int/float, optkit.Matrix)\nProvided:" 
				"\n\t({},{},{},{},{},{},{}). \n{}".format(
					type(tA),type(tB),type(alpha),type(A),
					type(B),type(beta),type(C), self.CHK_MSG))
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
			
		At = enums.CblasTrans if tA =='T' else enums.CblasNoTrans
		Bt = enums.CblasTrans if tB =='T' else enums.CblasNoTrans

		self.denselib.blas_gemm(self.dense_blas_handle, At, Bt, alpha, A.c, B.c, beta, C.c)


	def cholesky_factor(self, A, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck and not isinstance(A, self.Matrix):
			raise TypeError("optkit.kernels.linsys.cholesky_factor(A) defined"
			      "only when argument is of"
				  "type opkit.Matrix. {}".format(self.CHK_MSG))
		else:
			if dimcheck and A.size1 != A.size2:
				raise ValueError("optkit.kernels.linsys.cholesky_factor(A)"
					   "only defined for square matrices A"
					   "A: {}x{}".format(A.size1, A.size2))
			
			self.denselib.linalg_cholesky_decomp(self.dense_blas_handle, A.c)

	def cholesky_solve(self, L, x, typecheck=None, dimcheck=None):
		if typecheck is None: typecheck = self.typecheck
		if dimcheck is None: dimcheck = self.dimcheck
		if typecheck:
			if not isinstance(L, self.Matrix):
				raise TypeError("optkit.kernels.linsys.cholesky_solve(L, x) defined"
					  "only when first argument is of"
					  "type opkit.Matrix. {}".format(self.CHK_MSG))
			elif not isinstance(x, self.Vector):
				raise TypeError("optkit.kernels.linsys.cholesky_solve(L, x) defined"
					  "only when second argument is of"
					  "type opkit.Vector. {}".format(self.CHK_MSG))

		if dimcheck and (x.size != L.size2 or x.size != L.size2): 
			raise ValueError("Error: optkit.kernels.linsys.cholesky_solve---"
				   "incompatible dimensions for x:=inv(L) * x\n"
				   "L: {}x{}\nx: {}".format(L.size1, L.size2, x.size))
			return

		self.denselib.linalg_cholesky_svx(self.dense_blas_handle, L.c,x.c)



