from optkit.compat import *

import numpy as np
import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.enums import OKFunctionEnums
from optkit.libs.linsys import attach_base_ctypes, attach_dense_linsys_ctypes,\
	attach_sparse_linsys_ctypes, attach_base_ccalls, attach_vector_ccalls, \
	attach_dense_linsys_ccalls, attach_sparse_linsys_ccalls
from optkit.libs.prox import attach_prox_ctypes, attach_prox_ccalls
from optkit.libs.operator import attach_operator_ctypes, attach_operator_ccalls
from optkit.libs.cg import attach_cg_ctypes, attach_cg_ccalls
from optkit.libs.equilibration import attach_equilibration_ccalls, \
	attach_operator_equilibration_ccalls
from optkit.libs.projector import attach_projector_ctypes, \
	attach_projector_ccalls, attach_operator_projector_ctypes_ccalls

class PogsLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libpogs_dense_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_prox_ctypes)
		self.attach_calls.append(attach_prox_ccalls)
		self.attach_calls.append(attach_projector_ctypes)
		self.attach_calls.append(attach_projector_ccalls)
		self.attach_calls.append(attach_equilibration_ccalls)
		self.attach_calls.append(attach_pogs_common_ctypes)
		self.attach_calls.append(attach_pogs_common_ccalls)
		self.attach_calls.append(attach_pogs_ctypes)
		self.attach_calls.append(attach_pogs_ccalls)

class PogsIndirectLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libpogs_sparse_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_sparse_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_sparse_linsys_ccalls)
		self.attach_calls.append(attach_prox_ctypes)
		self.attach_calls.append(attach_prox_ccalls)
		self.attach_calls.append(attach_operator_ctypes)
		self.attach_calls.append(attach_operator_ccalls)
		self.attach_calls.append(attach_cg_ctypes)
		self.attach_calls.append(attach_cg_ccalls)
		self.attach_calls.append(attach_projector_ctypes)
		self.attach_calls.append(attach_projector_ccalls)
		self.attach_calls.append(attach_equilibration_ccalls)
		self.attach_calls.append(attach_operator_equilibration_ccalls)
		self.attach_calls.append(attach_pogs_common_ctypes)
		self.attach_calls.append(attach_pogs_common_ccalls)
		self.attach_calls.append(attach_pogs_ctypes)
		self.attach_calls.append(attach_pogs_ccalls)

class PogsAbstractLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libpogs_abstract_')
		self.attach_calls.append(attach_base_ctypes)
		self.attach_calls.append(attach_dense_linsys_ctypes)
		self.attach_calls.append(attach_sparse_linsys_ctypes)
		self.attach_calls.append(attach_base_ccalls)
		self.attach_calls.append(attach_vector_ccalls)
		self.attach_calls.append(attach_dense_linsys_ccalls)
		self.attach_calls.append(attach_sparse_linsys_ccalls)
		self.attach_calls.append(attach_prox_ctypes)
		self.attach_calls.append(attach_prox_ccalls)
		self.attach_calls.append(attach_operator_ctypes)
		self.attach_calls.append(attach_operator_ccalls)
		self.attach_calls.append(attach_cg_ctypes)
		self.attach_calls.append(attach_cg_ccalls)
		self.attach_calls.append(attach_projector_ctypes)
		self.attach_calls.append(attach_projector_ccalls)
		self.attach_calls.append(attach_operator_projector_ctypes_ccalls)
		self.attach_calls.append(attach_equilibration_ccalls)
		self.attach_calls.append(attach_operator_equilibration_ccalls)
		self.attach_calls.append(attach_pogs_common_ctypes)
		self.attach_calls.append(attach_pogs_common_ccalls)
		self.attach_calls.append(attach_pogs_abstract_ctypes)
		self.attach_calls.append(attach_pogs_abstract_ccalls)

def attach_pogs_common_ctypes(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p

	# lib properties
	lib.private_api_accessible.restype = ct.c_int
	lib.full_api_accessible = lib.private_api_accessible()

	# Public API
	class PogsSettings(ct.Structure):
		_fields_ = [('alpha', ok_float),
					('rho', ok_float),
					('abstol', ok_float),
					('reltol', ok_float),
					('maxiter', ct.c_uint),
					('verbose', ct.c_uint),
					('suppress', ct.c_uint),
					('adaptiverho', ct.c_int),
					('gapstop', ct.c_int),
					('warmstart', ct.c_int),
					('resume', ct.c_int),
					('x0', ok_float_p),
					('nu0', ok_float_p)]

	lib.pogs_settings = PogsSettings
	lib.pogs_settings_p = ct.POINTER(lib.pogs_settings)

	class PogsInfo(ct.Structure):
		_fields_ = [('err', ct.c_int),
					('converged', ct.c_int),
					('k', ct.c_uint),
					('obj', ok_float),
					('rho', ok_float),
					('setup_time', ok_float),
					('solve_time', ok_float)]
		def __init__(self):
			self.err = 0
			self.converged = 0
			self.k = 0
			self.obj = np.nan
			self.rho = np.nan
			self.setup_time = np.nan
			self.solve_time = np.nan

	lib.pogs_info = PogsInfo
	lib.pogs_info_p =  ct.POINTER(lib.pogs_info)

	class PogsOutput(ct.Structure):
		_fields_ = [('x', ok_float_p),
					('y', ok_float_p),
					('mu', ok_float_p),
					('nu', ok_float_p)]

	lib.pogs_output = PogsOutput
	lib.pogs_output_p =  ct.POINTER(lib.pogs_output)

	class AdaptiveRhoParameters(ct.Structure):
		_fields_ = [('delta', ok_float),
					('l', ok_float),
					('u', ok_float),
					('xi', ok_float)]

	lib.adapt_params = AdaptiveRhoParameters
	lib.adapt_params_p = ct.POINTER(lib.adapt_params)

	class PogsBlockVector(ct.Structure):
		_fields_ = [('size', ct.c_size_t),
					('m', ct.c_size_t),
					('n', ct.c_size_t),
					('x', vector_p),
					('y', vector_p),
					('vec', vector_p)]

	lib.block_vector = PogsBlockVector
	lib.block_vector_p = ct.POINTER(lib.block_vector)
	block_vector_p = lib.block_vector_p

	class PogsResiduals(ct.Structure):
		_fields_ = [('primal', ok_float),
					('dual', ok_float),
					('gap', ok_float)]
		def __init__(self):
			self.primal = np.nan
			self.dual = np.nan
			self.gap = np.nan

	lib.pogs_residuals = PogsResiduals
	lib.pogs_residuals_p = ct.POINTER(lib.pogs_residuals)

	class PogsTolerances(ct.Structure):
		_fields_ = [('primal', ok_float),
					('dual', ok_float),
					('gap', ok_float),
					('reltol', ok_float),
					('abstol', ok_float),
					('atolm', ok_float),
					('atoln', ok_float),
					('atolmn', ok_float)]

	lib.pogs_tolerances = PogsTolerances
	lib.pogs_tolerances_p = ct.POINTER(lib.pogs_tolerances)

	class PogsObjectives(ct.Structure):
		_fields_ = [('primal', ok_float),
					('dual', ok_float),
					('gap', ok_float)]
		def __init__(self):
			self.primal = np.nan
			self.dual = np.nan
			self.gap = np.nan

	lib.pogs_objectives = PogsObjectives
	lib.pogs_objectives_p = ct.POINTER(lib.pogs_objectives)

	class PogsVariables(ct.Structure):
		_fields_ = [('primal', block_vector_p),
					('primal12', block_vector_p),
					('dual', block_vector_p),
					('dual12', block_vector_p),
					('prev', block_vector_p),
					('temp', block_vector_p),
					('m', ct.c_size_t),
					('n', ct.c_size_t)]

	lib.pogs_variables = PogsVariables
	lib.pogs_variables_p = ct.POINTER(lib.pogs_variables)

def attach_pogs_ctypes(lib, single_precision=False):
	if not 'matrix_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'function_vector_p' in lib.__dict__:
		attach_prox_ctypes(lib, single_precision)
	if not 'pogs_settings_p' in lib.__dict__:
		attach_pogs_common_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	matrix_p = lib.matrix_p
	function_vector_p = lib.function_vector_p
	pogs_settings_p = lib.pogs_settings_p
	pogs_variables_p = lib.pogs_variables_p

	# lib properties
	full_api_accessible = lib.full_api_accessible
	lib.is_direct.restype = ct.c_int
	lib.direct = lib.is_direct()

	class PogsMatrix(ct.Structure):
		_fields_ = [('A', matrix_p),
					('P', ct.c_void_p),
					('d', vector_p),
					('e', vector_p),
					('normA', ok_float),
					('skinny', ct.c_int),
					('normalized', ct.c_int),
					('equilibrated', ct.c_int)]

	lib.pogs_matrix = PogsMatrix
	lib.pogs_matrix_p = ct.POINTER(lib.pogs_matrix)
	pogs_matrix_p = lib.pogs_matrix_p

	class PogsSolver(ct.Structure):
		_fields_ = [('M', pogs_matrix_p),
					('z', pogs_variables_p),
					('f', function_vector_p),
					('g', function_vector_p),
					('rho', ok_float),
					('settings', pogs_settings_p),
					('linalg_handle', ct.c_void_p),
					('init_time', ok_float)]

	lib.pogs_solver = PogsSolver
	lib.pogs_solver_p = ct.POINTER(lib.pogs_solver)

def attach_pogs_common_ccalls(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'function_vector_p' in lib.__dict__:
		attach_prox_ctypes(lib, single_precision)
	if not 'pogs_variables_p' in lib.__dict__:
		attach_pogs_common_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	function_vector_p = lib.function_vector_p

	pogs_output_p = lib.pogs_output_p
	pogs_info_p = lib.pogs_info_p
	pogs_settings_p = lib.pogs_settings_p
	adapt_params_p = lib.adapt_params_p
	block_vector_p = lib.block_vector_p
	pogs_residuals_p = lib.pogs_residuals_p
	pogs_tolerances = lib.pogs_tolerances
	pogs_tolerances_p = lib.pogs_tolerances_p
	pogs_objectives_p = lib.pogs_objectives_p
	pogs_variables_p = lib.pogs_variables_p

	lib.set_default_settings.argtypes = [pogs_settings_p]
	lib.set_default_settings.restype = ct.c_uint

	# Private API
	if lib.full_api_accessible:
		## argtypes
		lib.initialize_conditions.argtypes = [
				pogs_objectives_p, pogs_residuals_p, pogs_tolerances_p,
				pogs_settings_p, ct.c_size_t, ct.c_size_t]
		lib.set_prev.argtypes = [pogs_variables_p]
		lib.prox.argtypes = [
				ct.c_void_p, function_vector_p, function_vector_p,
				pogs_variables_p, ok_float]
		lib.update_dual.argtypes = [ct.c_void_p, pogs_variables_p, ok_float]
		lib.adaptrho.argtypes = [
				pogs_variables_p, pogs_settings_p, ok_float_p, adapt_params_p,
				pogs_residuals_p, pogs_tolerances_p, ct.c_uint]
		lib.copy_output.argtypes = [
				pogs_output_p, pogs_variables_p, vector_p, vector_p, ok_float,
				ct.c_uint]

		## results
		lib.initialize_conditions.restype = ct.c_uint
		lib.set_prev.restype = ct.c_uint
		lib.prox.restype = ct.c_uint
		lib.update_dual.restype = ct.c_uint
		lib.adaptrho.restype = ct.c_uint
		lib.copy_output.restype = ct.c_uint
	else:
		lib.initialize_conditions = AttributeError()
		lib.set_prev = AttributeError()
		lib.prox = AttributeError()
		lib.update_dual = AttributeError()
		lib.adaptrho = AttributeError()
		lib.copy_output = AttributeError()

def attach_pogs_ccalls(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'function_vector_p' in lib.__dict__:
		attach_prox_ctypes(lib, single_precision)
	if not 'pogs_solver_p' in lib.__dict__:
		attach_pogs_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	vector_p = lib.vector_p
	function_vector_p = lib.function_vector_p

	pogs_output_p = lib.pogs_output_p
	pogs_info_p = lib.pogs_info_p
	pogs_settings_p = lib.pogs_settings_p
	adapt_params_p = lib.adapt_params_p
	block_vector_p = lib.block_vector_p
	pogs_residuals_p = lib.pogs_residuals_p
	pogs_tolerances = lib.pogs_tolerances
	pogs_tolerances_p = lib.pogs_tolerances_p
	pogs_objectives_p = lib.pogs_objectives_p
	pogs_matrix_p = lib.pogs_matrix_p
	pogs_variables_p = lib.pogs_variables_p
	pogs_solver_p = lib.pogs_solver_p

	## arguments
	lib.pogs_init.argtypes = [ok_float_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.pogs_solve.argtypes = [
			ct.c_void_p, function_vector_p, function_vector_p, pogs_settings_p,
			pogs_info_p, pogs_output_p]
	lib.pogs_finish.argtypes = [ct.c_void_p, ct.c_int]
	lib.pogs.argtypes = [
			ok_float_p, function_vector_p, function_vector_p, pogs_settings_p,
			pogs_info_p, pogs_output_p, ct.c_uint, ct.c_int]
	lib.pogs_load_solver.argtypes = [
			ok_float_p, ok_float_p, ok_float_p, ok_float_p, ok_float_p,
			ok_float_p, ok_float_p, ok_float_p, ok_float_p, ok_float,
			ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.pogs_extract_solver.argtypes = [
			ct.c_void_p, ok_float_p, ok_float_p, ok_float_p, ok_float_p,
			ok_float_p, ok_float_p, ok_float_p, ok_float_p, ok_float_p,
			ok_float_p, ct.c_uint]
	lib.pogs_solver_exists.argtypes = [pogs_solver_p]


	## return types
	lib.pogs_init.restype = pogs_solver_p
	lib.pogs_solve.restype = ct.c_uint
	lib.pogs_finish.restype = ct.c_uint
	lib.pogs.restype = ct.c_uint
	lib.pogs_load_solver.restype = pogs_solver_p
	lib.pogs_extract_solver.restype = ct.c_uint
	lib.pogs_solver_exists.restype = ct.c_uint

	# Private API
	if lib.full_api_accessible:
		## argtypes
		lib.update_problem.argtypes = [
				pogs_solver_p, function_vector_p, function_vector_p]
		lib.initialize_variables.argtypes = [pogs_solver_p]
		lib.pogs_solver_loop.argtypes = [pogs_solver_p, pogs_info_p]
		lib.project_primal.argtypes = [
				ct.c_void_p, ct.c_void_p, pogs_variables_p, ok_float]
		lib.check_convergence.argtypes = [
				ct.c_void_p, pogs_solver_p, pogs_objectives_p,
				pogs_residuals_p, pogs_tolerances_p]

		## results
		lib.update_problem.restype = ct.c_uint
		lib.initialize_variables.restype = ct.c_uint
		lib.pogs_solver_loop.restype = ct.c_uint
		lib.project_primal.restype = ct.c_uint
		lib.check_convergence.restype = ct.c_int

		proj_argtypes = [
				ct.c_void_p, ct.c_void_p, vector_p, vector_p, vector_p,
				vector_p]
		proj_restype = ct.c_uint

		if lib.direct:
			lib.direct_projector_project.argtypes = proj_argtypes
			lib.direct_projector_project.restype = proj_restype
		else:
			lib.indirect_projector_project.argtypes = proj_argtypes
			lib.indirect_projector_project.restype = proj_restype
	else:
		lib.update_problem = AttributeError()
		lib.initialize_variables = AttributeError()
		lib.pogs_solver_loop = AttributeError()
		lib.project_primal = AttributeError()
		lib.check_convergence = AttributeError()
		lib.direct_projector_project = AttributeError()
		lib.indirect_projector_project = AttributeError()

def attach_pogs_abstract_ctypes(lib, single_precision=False):
	if not 'vector_p' in lib.__dict__:
		attach_dense_linsys_ctypes(lib, single_precision)
	if not 'function_vector_p' in lib.__dict__:
		attach_prox_ctypes(lib, single_precision)
	if not 'operator_p' in lib.__dict__:
		attach_operator_ctypes(lib, single_precision)
	if not 'projector_p' in lib.__dict__:
		attach_projector_ctypes(lib, single_precision)
	if not 'pogs_settings_p' in lib.__dict__:
		attach_pogs_common_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	vector_p = lib.vector_p
	function_vector_p = lib.function_vector_p
	operator_p = lib.operator_p
	projector_p = lib.projector_p
	pogs_settings_p = lib.pogs_settings_p
	pogs_variables_p = lib.pogs_variables_p

	# lib properties
	lib.private_api_accessible.restype = ct.c_int
	lib.full_api_accessible = lib.private_api_accessible()

	class PogsWork(ct.Structure):
		_fields_ = [('A', operator_p),
					('P', projector_p),
					('d', vector_p),
					('e', vector_p),
					('operator_scale', ct.CFUNCTYPE(
							ct.c_uint, operator_p, ok_float)),
					('operator_equilibrate', ct.CFUNCTYPE(
							ct.c_uint, ct.c_void_p, operator_p, vector_p,
							vector_p, ok_float)),
					('normA', ok_float),
					('skinny', ct.c_int),
					('normalized', ct.c_int),
					('equilibrated', ct.c_int)]

	lib.pogs_work = PogsWork
	lib.pogs_work_p = ct.POINTER(lib.pogs_work)

	class PogsSolver(ct.Structure):
		_fields_ = [('W', lib.pogs_work_p),
					('z', pogs_variables_p),
					('f', function_vector_p),
					('g', function_vector_p),
					('rho', ok_float),
					('settings', pogs_settings_p),
					('linalg_handle', ct.c_void_p),
					('init_time', ok_float)]

	lib.pogs_solver = PogsSolver
	lib.pogs_solver_p = ct.POINTER(lib.pogs_solver)

def attach_pogs_abstract_ccalls(lib, single_precision=False):
	if not 'pogs_work_p' in lib.__dict__:
		attach_pogs_abstract_ctypes(lib, single_precision)

	ok_float = lib.ok_float
	ok_float_p = lib.ok_float_p
	ok_int = lib.ok_int
	ok_int_p = lib.ok_int_p
	vector_p = lib.vector_p
	function_vector_p = lib.function_vector_p
	operator_p = lib.operator_p
	pogs_settings_p = lib.pogs_settings_p
	pogs_info_p = lib.pogs_info_p
	pogs_output_p = lib.pogs_output_p
	pogs_variables_p = lib.pogs_variables_p
	pogs_solver_p = lib.pogs_solver_p
	pogs_work_p = lib.pogs_work_p

	## arguments
	lib.pogs_init.argtypes = [operator_p, ct.c_int, ok_float]
	lib.pogs_solve.argtypes = [
			pogs_solver_p, function_vector_p, function_vector_p,
			pogs_settings_p, pogs_info_p, pogs_output_p]
	lib.pogs_finish.argtypes = [pogs_solver_p, ct.c_int]
	lib.pogs.argtypes = [
			operator_p, function_vector_p, function_vector_p, pogs_settings_p,
			pogs_info_p, pogs_output_p, ct.c_int, ok_float, ct.c_int]
	lib.pogs_dense_operator_gen.argtypes = [
			ok_float_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.pogs_sparse_operator_gen.argtypes = [
			ok_float_p, ok_int_p, ok_int_p, ct.c_size_t, ct.c_size_t,
			ct.c_size_t, ct.c_uint]
	lib.pogs_dense_operator_free.argtypes = [operator_p]
	lib.pogs_sparse_operator_free.argtypes = [operator_p]

	# lib.pogs_load_solver.argtypes = [ok_float_p, ok_float_p,
	# 								 ok_float_p, ok_float_p,
	# 								 ok_float_p, ok_float_p,
	# 								 ok_float_p, ok_float_p,
	# 								 ok_float_p, ok_float, ct.c_size_t,
	# 								 ct.c_size_t, ct.c_uint]
	# lib.pogs_extract_solver.argtypes = [ct.c_void_p, ok_float_p,
	# 									ok_float_p, ok_float_p,
	# 									ok_float_p, ok_float_p,
	# 									ok_float_p, ok_float_p,
	# 									ok_float_p, ok_float_p,
	# 									ok_float_p, ct.c_uint]

	## return types
	lib.pogs_init.restype = pogs_solver_p
	lib.pogs_solve.restype = ct.c_uint
	lib.pogs_finish.restype = ct.c_uint
	lib.pogs.restype = ct.c_uint
	lib.pogs_dense_operator_gen.restype = operator_p
	lib.pogs_sparse_operator_gen.restype = operator_p
	lib.pogs_dense_operator_free.restype = ct.c_uint
	lib.pogs_sparse_operator_free.restype = ct.c_uint

	# lib.pogs_load_solver.restype = ct.c_void_p
	# lib.pogs_extract_solver.restype = ct.c_uint

	# Private API
	if lib.full_api_accessible:
		projector_p = lib.projector_p
		pogs_residuals_p = lib.pogs_residuals_p
		pogs_tolerances = lib.pogs_tolerances
		pogs_tolerances_p = lib.pogs_tolerances_p
		pogs_objectives_p = lib.pogs_objectives_p

		## argtypes
		lib.update_problem.argtypes = [
				pogs_solver_p,function_vector_p, function_vector_p]
		lib.initialize_variables.argtypes = [pogs_solver_p]
		lib.pogs_solver_loop.argtypes = [pogs_solver_p, pogs_info_p]
		lib.project_primal.argtypes = [
				ct.c_void_p, projector_p, pogs_variables_p, ok_float, ok_float]
		lib.check_convergence.argtypes = [
				ct.c_void_p, pogs_solver_p, pogs_objectives_p,
				pogs_residuals_p, pogs_tolerances_p]

		## results
		lib.update_problem.restype = ct.c_uint
		lib.initialize_variables.restype = ct.c_uint
		lib.pogs_solver_loop.restype = ct.c_uint
		lib.project_primal.restype = ct.c_uint
		lib.check_convergence.restype = ct.c_int

		# redefinitions
		# lib.pogs_init.restype = pogs_solver_p
		# lib.pogs_load_solver.restype = pogs_solver_p

	else:
		lib.update_problem = AttributeError()
		lib.initialize_variables = AttributeError()
		lib.pogs_solver_loop = AttributeError()
		lib.project_primal = AttributeError()
		lib.check_convergence = AttributeError()