from optkit.backends import OKBackend
from optkit.types.highlevel import HighLevelLinsysTypes, \
	HighLevelProxTypes, HighLevelPogsTypes
from optkit.py_implementations.kernels import LinsysCoreKernels, \
	LinsysExtensionKernels, ProxKernels
from optkit.py_implementations.projector import DirectProjectorFactory
from optkit.py_implementations.equilibration import EquilibrationMethods
from optkit.py_implementations.pogs import POGSDirectSolver
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
Python implementations
"""
linsys = None
prox = None
DirectProjector = None
equil = None
pogs = None

"""
C implementations
"""
CPogsTypes = None
PogsSolver = None
PogsObjective = None

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

	## Python implementations
	# Linsys calls
	global linsys

	# Prox calls
	global prox

	# Projector 
	global DirectProjector

	# Equilibration methods
	global equil

	# Solver methods
	global pogs

	## C implementations
	global DirectProjector
	global CPogsTypes
	global PogsSolver
	global PogsObjective

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
	linsys = {}
	linsys['set_all'] = linsys_core_kernels.set_all	
	linsys['copy'] = linsys_core_kernels.copy
	linsys['view'] = linsys_core_kernels.view
	linsys['sync'] = linsys_core_kernels.sync
	linsys['print_var'] = linsys_core_kernels.print_var
	linsys['add'] = linsys_core_kernels.add
	linsys['sub'] = linsys_core_kernels.sub
	linsys['mul'] = linsys_core_kernels.mul
	linsys['div'] = linsys_core_kernels.div
	linsys['elemwise_inverse'] = linsys_core_kernels.elemwise_inverse
	linsys['elemwise_sqrt'] = linsys_core_kernels.elemwise_sqrt
	linsys['elemwise_inverse_sqrt'] = linsys_core_kernels.elemwise_inverse_sqrt
	linsys['dot'] = linsys_core_kernels.dot
	linsys['asum'] = linsys_core_kernels.asum
	linsys['nrm2'] = linsys_core_kernels.nrm2
	linsys['axpy'] = linsys_core_kernels.axpy
	linsys['gemv'] = linsys_core_kernels.gemv
	linsys['gemm'] = linsys_core_kernels.gemm
	linsys['cholesky_factor'] = linsys_core_kernels.cholesky_factor
	linsys['cholesky_solve'] = linsys_core_kernels.cholesky_solve

	# (reset extensions)
	linsys['splitview'] = linsys_extensions.splitview
	linsys['axpby'] = linsys_extensions.axpby
	linsys['axpby_inplace'] = linsys_extensions.axpby_inplace
	linsys['diag'] = linsys_extensions.diag
	linsys['aidpm'] = linsys_extensions.aidpm
	linsys['add_diag'] = linsys_extensions.add_diag
	linsys['sum_diag'] = linsys_extensions.sum_diag
	linsys['norm_diag'] = linsys_extensions.norm_diag
	linsys['mean_diag'] = linsys_extensions.mean_diag
	linsys['gramian'] = linsys_extensions.gramian
	linsys['get_curryable_gemv'] = linsys_extensions.get_curryable_gemv

	# reset prox calls
	prox = {}
	prox['scale_function_vector'] = prox_kernels.scale_function_vector
	prox['push_function_vector'] = prox_kernels.push_function_vector
	prox['print_function_vector'] = prox_kernels.print_function_vector
	prox['func_eval'] = prox_kernels.eval
	prox['prox_eval'] = prox_kernels.prox

	# single dictionary with all linsys and prox calls, for internal use.
	base_kernels = linsys.copy()
	base_kernels.update(prox)

	# (projection)
	DirectProjector = DirectProjectorFactory(
		base_kernels, Matrix).DirectProjector 

	# (equilibration)
	equil_methods = EquilibrationMethods(base_kernels, Vector, Matrix)
	equil = {}
	equil['dense_l2'] = equil_methods.dense_l2
	equil['sinkhornknopp'] = equil_methods.sinkhornknopp

	# (solver)
	pogs = POGSDirectSolver(backend, base_kernels, Vector, Matrix, 
		FunctionVector, DirectProjector, equil_methods)


	## C implemenetations
	CPogsTypes = HighLevelPogsTypes(backend)
	PogsSolver = CPogsTypes.Solver
	PogsObjective = CPogsTypes.Objective

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


