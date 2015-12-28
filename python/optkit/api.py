from optkit.backends import OKBackend
from optkit.types.highlevel import HighLevelLinsysTypes, HighLevelProxTypes, \
	HighLevelPogsTypes
from optkit.kernels import LinsysCoreKernels, LinsysExtensionKernels, \
	ProxKernels
from optkit.projector import DirectProjectorFactory
from optkit.equilibration import EquilibrationMethods
from optkit.pogs import POGSDirectSolver
from os import getenv

"""
Backend handle
"""
backend = OKBackend()

"""
Types
"""
Vector = Matrix = FunctionVector = None

"""
Linsys calls
"""
# Core: array operations
set_all = copy = view = sync = print_var = None

# Core: elementwise vector operations
add = sub = mul = div = None
elemwise_inverse = elemwise_sqrt = elemwise_inverse_sqrt = None 

# Core: blas level 1 operations
dot = asum = nrm2 = axpy = None

# Core: blas level 2/3 & lapack operations
gemv = gemm = cholesky_factor = cholesky_solve = None

# Extensions: x = [x1 x2]
splitview = None

# Extensions: z = ax + by; y = ax + by
axpby = axpby_inplace = None

# Extensions: diag(A); A += aI; sum(diag(A)); norm(diag(A), [l_1 ,l_2]) 
diag = aidpm = add_diag = sum_diag = norm_diag = mean_diag = None

# Extensions: B = A'A or AA'; mul(A,'tA') = gemv('tA', a, A, x, b, y)
gramian = get_curryable_gemv = None


"""
Prox calls
"""
scale_function_vector = push_function_vector = print_function_vector = None
func_eval = prox_eval = None

"""
Projectors
"""
DirectProjector = None
DirectProjectorPy = None


"""
Equilibration methods
"""
dense_l2_equilibration = sinkhornknopp_equilibration = None


"""
Solver types and methods
"""
pogs = None
CPogsTypes = None
PogsSolver = None

"""
Backend switching
"""
def set_backend(GPU=False, double=True, force_rowmajor=False,
	force_colmajor=False):
	# Backend
	global backend

	# Types
	global Vector
	global Matrix
	global FunctionVector

	# Linsys calls
	global set_all
	global copy
	global view
	global sync
	global print_var
	global add
	global sub
	global mul
	global div
	global elemwise_inverse
	global elemwise_sqrt
	global elemwise_inverse_sqrt
	global dot
	global asum
	global nrm2
	global axpy
	global gemv
	global gemm
	global cholesky_factor
	global cholesky_solve

	global splitview
	global axpby
	global axpby_inplace
	global diag
	global aidpm
	global add_diag
	global sum_diag
	global norm_diag
	global mean_diag
	global gramian
	global get_curryable_gemv

	# Prox calls
	global scale_function_vector
	global push_function_vector
	global print_function_vector
	global func_eval
	global prox_eval

	# Projector types
	global DirectProjector
	global DirectProjectorPy

	# Equilibration methods
	global dense_l2_equilibration
	global sinkhornknopp_equilibration

	# Solver methods
	global pogs
	global CPogsTypes
	global PogsSolver

	# change backend
	backend_name=backend.change(GPU=GPU, double=double, 
		force_rowmajor=force_rowmajor, force_colmajor=force_colmajor)
	

	# reset types
	linsys_type_factory = HighLevelLinsysTypes(backend)
	prox_type_factory = HighLevelProxTypes(backend)

	Vector = linsys_type_factory.Vector
	Matrix = linsys_type_factory.Matrix
	FunctionVector = prox_type_factory.FunctionVector

	linsys_core_kernels = LinsysCoreKernels(backend, Vector, Matrix)
	linsys_extensions = LinsysExtensionKernels(linsys_core_kernels, Matrix)
	prox_kernels = ProxKernels(backend, Vector, FunctionVector)

	# reset linsys calls
	# (reset core)
	set_all = linsys_core_kernels.set_all	
	copy = linsys_core_kernels.copy
	view = linsys_core_kernels.view
	sync = linsys_core_kernels.sync
	print_var = linsys_core_kernels.print_var
	add = linsys_core_kernels.add
	sub = linsys_core_kernels.sub
	mul = linsys_core_kernels.mul
	div = linsys_core_kernels.div
	elemwise_inverse = linsys_core_kernels.elemwise_inverse
	elemwise_sqrt = linsys_core_kernels.elemwise_sqrt
	elemwise_inverse_sqrt = linsys_core_kernels.elemwise_inverse_sqrt
	dot = linsys_core_kernels.dot
	asum = linsys_core_kernels.asum
	nrm2 = linsys_core_kernels.nrm2
	axpy = linsys_core_kernels.axpy
	gemv = linsys_core_kernels.gemv
	gemm = linsys_core_kernels.gemm
	cholesky_factor = linsys_core_kernels.cholesky_factor
	cholesky_solve = linsys_core_kernels.cholesky_solve

	# (reset extensions)
	splitview = linsys_extensions.splitview
	axpby = linsys_extensions.axpby
	axpby_inplace = linsys_extensions.axpby_inplace
	diag = linsys_extensions.diag
	aidpm = linsys_extensions.aidpm
	add_diag = linsys_extensions.add_diag
	sum_diag = linsys_extensions.sum_diag
	norm_diag = linsys_extensions.norm_diag
	mean_diag = linsys_extensions.mean_diag
	gramian = linsys_extensions.gramian
	get_curryable_gemv = linsys_extensions.get_curryable_gemv

	# reset prox calls
	scale_function_vector = prox_kernels.scale_function_vector
	push_function_vector = prox_kernels.push_function_vector
	print_function_vector = prox_kernels.print_function_vector
	func_eval = prox_kernels.eval
	prox_eval = prox_kernels.prox

	base_kernels = {'set_all': set_all, 'copy': copy, 'view': view, 'sync':
		sync, 'print_var': print_var, 'add': add, 'sub': sub, 'mul': mul,
		'div': div, 'elemwise_inverse': elemwise_inverse, 'elemwise_sqrt':
		elemwise_sqrt, 'elemwise_inverse_sqrt': elemwise_inverse_sqrt,
		'dot': dot, 'asum': asum, 'nrm2': nrm2, 'axpy': axpy, 'gemv':
		gemv, 'gemm': gemm, 'cholesky_factor': cholesky_factor, 'cholesky_solve':
		cholesky_solve, 'splitview': splitview, 'axpby': axpby,
		'axpby_inplace': axpby_inplace, 'diag': diag, 'aidpm': aidpm,
		'add_diag': add_diag, 'sum_diag': sum_diag, 'norm_diag': norm_diag,
		'mean_diag': mean_diag, 'gramian': gramian, 'get_curryable_gemv':
		get_curryable_gemv, 'scale_function_vector': scale_function_vector,
		'push_function_vector': push_function_vector, 'print_function_vector':
		print_function_vector, 'func_eval': func_eval, 'prox_eval': prox_eval}


	# reset algorithmic types
	# (projection)
	DirectProjectorPy = DirectProjectorFactory(
		base_kernels, Matrix).DirectProjector 

	# (equilibration)
	equil_methods = EquilibrationMethods(base_kernels, Vector, Matrix)
	dense_l2_equilibration = equil_methods.dense_l2
	sinkhornknopp_equilibration = equil_methods.sinkhornknopp

	# (solver)
	pogs = POGSDirectSolver(backend, base_kernels, Vector, Matrix, 
		FunctionVector, DirectProjectorPy, equil_methods)

	CPogsTypes = HighLevelPogsTypes(backend, FunctionVector)
	PogsSolver = CPogsTypes.Solver

	print "optkit backend set to {}".format(backend.libname)


"""
INITIALIZATION BEHAVIOR:
"""


default_device = getenv('OPTKIT_DEFAULT_DEVICE', 'cpu')
default_precision = getenv('OPTKIT_DEFAULT_FLOATBITS', '64')
default_order = getenv('OPTKIT_DEFAULT_ORDER', '')


set_backend(GPU=(default_device == 'gpu'), 
	double=(default_precision == '64'),
	force_rowmajor=(default_order == 'row'),
	force_colmajor=(default_order == 'col'))


