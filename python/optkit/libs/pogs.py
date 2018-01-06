from optkit.compat import *

import numpy as np
import ctypes as ct

from optkit.libs.loader import OptkitLibs
from optkit.libs.enums import OKEnums, OKFunctionEnums
from optkit.libs.linsys import include_ok_dense
from optkit.libs.prox import include_ok_prox, ok_prox_API
from optkit.libs.operator import include_ok_operator
from optkit.libs.equilibration import ok_equil_dense_API, ok_equil_API
from optkit.libs.projector import include_ok_projector, \
		ok_projector_dense_API, ok_projector_API
from optkit.libs.anderson import include_ok_anderson, ok_anderson_API

def include_ok_pogs_datatypes(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'pogs_variables_p', attach_pogs_datatypes_ctypes, **include_args)

def include_ok_pogs_adaptrho(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'adapt_params_p', attach_pogs_adaptrho_ctypes, **include_args)

def include_ok_pogs(lib, **include_args):
	OptkitLibs.conditional_include(
		lib, 'pogs_solver_p', attach_pogs_ctypes, **include_args)

def ok_pogs_common_API(): return (
		ok_prox_API()
		+ ok_anderson_API()
		+ [attach_pogs_ccalls]
	)

def ok_pogs_dense_API(): return (
		ok_pogs_common_API()
		+ ok_projector_dense_API()
		+ ok_equil_dense_API()
	)

#def ok_pogs_sparse_API(): return []

def ok_pogs_abstract_API(): return (
		ok_pogs_common_API()
		+ ok_projector_API()
		+ ok_equil_API()
	)

class PogsDenseLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libpogs_dense_', ok_pogs_dense_API())

# class PogsIndirectLibs(OptkitLibs):
# 	def __init__(self):
# 		OptkitLibs.__init__(self, 'libpogs_sparse_', ok_pogs_sparse_API())

class PogsAbstractLibs(OptkitLibs):
	def __init__(self):
		OptkitLibs.__init__(self, 'libpogs_abstract_', ok_pogs_abstract_API())

def set_pogs_impl(lib):
	enums = OKEnums()
	lib.get_pogs_impl.restype = ct.c_uint

	pogs_impl = lib.get_pogs_impl()
	if pogs_impl == enums.OkPogsAbstract:
		lib.py_pogs_impl = 'abstract'
	elif pogs_impl == enums.OkPogsSparse:
		raise NotImplementedError('POGS sparse not implemented')
		# lib.py_pogs_impl = 'sparse'
	elif pogs_impl == enums.OkPogsDense:
		lib.py_pogs_impl = 'dense'
		lib.direct = True
	else:
		raise ValueError('unknown POGS implementation')

def attach_pogs_datatypes_ctypes(lib, single_precision=False):
	include_ok_dense(lib, single_precision=single_precision)

	class PogsGraphVector(ct.Structure):
		_fields_ = [('size', ct.c_size_t),
					('m', ct.c_size_t),
					('n', ct.c_size_t),
					('x', lib.vector_p),
					('y', lib.vector_p),
					('vec', lib.vector_p),
					('memory_attached', ct.c_int)]

	lib.graph_vector = PogsGraphVector
	lib.graph_vector_p = ct.POINTER(lib.graph_vector)
	graph_vector_p = lib.graph_vector_p

	class PogsResiduals(ct.Structure):
		_fields_ = [('primal', lib.ok_float),
					('dual', lib.ok_float),
					('gap', lib.ok_float)]
		def __init__(self):
			self.primal = np.nan
			self.dual = np.nan
			self.gap = np.nan

	lib.pogs_residuals = PogsResiduals
	lib.pogs_residuals_p = ct.POINTER(lib.pogs_residuals)

	class PogsTolerances(ct.Structure):
		_fields_ = [('primal', lib.ok_float),
					('dual', lib.ok_float),
					('gap', lib.ok_float),
					('reltol', lib.ok_float),
					('abstol', lib.ok_float),
					('atolm', lib.ok_float),
					('atoln', lib.ok_float),
					('atolmn', lib.ok_float)]

	lib.pogs_tolerances = PogsTolerances
	lib.pogs_tolerances_p = ct.POINTER(lib.pogs_tolerances)

	class PogsObjectiveValues(ct.Structure):
		_fields_ = [('primal', lib.ok_float),
					('dual', lib.ok_float),
					('gap', lib.ok_float)]
		def __init__(self):
			self.primal = np.nan
			self.dual = np.nan
			self.gap = np.nan

	lib.pogs_objective_values = PogsObjectiveValues
	lib.pogs_objective_values_p = ct.POINTER(lib.pogs_objective_values)

	class PogsSettings(ct.Structure):
		_fields_ = [('alpha', lib.ok_float),
					('rho', lib.ok_float),
					('abstol', lib.ok_float),
					('reltol', lib.ok_float),
					('tolproj', lib.ok_float),
					('toladapt', lib.ok_float),
					('anderson_regularization', lib.ok_float),
					('maxiter', ct.c_uint),
					('anderson_lookback', ct.c_uint),
					('verbose', ct.c_uint),
					('suppress', ct.c_uint),
					('adaptiverho', ct.c_int),
					('accelerate', ct.c_int),
					('gapstop', ct.c_int),
					('warmstart', ct.c_int),
					('resume', ct.c_int),
					('diagnostic', ct.c_int),
					('x0', lib.ok_float_p),
					('nu0', lib.ok_float_p)]

	lib.pogs_settings = PogsSettings
	lib.pogs_settings_p = ct.POINTER(lib.pogs_settings)

	class PogsInfo(ct.Structure):
		_fields_ = [('err', ct.c_int),
					('converged', ct.c_int),
					('k', ct.c_uint),
					('obj', lib.ok_float),
					('rho', lib.ok_float),
					('setup_time', lib.ok_float),
					('solve_time', lib.ok_float)]
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
		_fields_ = [('x', lib.ok_float_p),
					('y', lib.ok_float_p),
					('mu', lib.ok_float_p),
					('nu', lib.ok_float_p),
					('primal_residuals', lib.ok_float_p),
					('dual_residuals', lib.ok_float_p),
					('primal_tolerances', lib.ok_float_p),
					('dual_tolerances', lib.ok_float_p)]

	lib.pogs_output = PogsOutput
	lib.pogs_output_p =  ct.POINTER(lib.pogs_output)

	class PogsVariables(ct.Structure):
		_fields_ = [('state', lib.vector_p),
					('fixed_point_iterate', lib.vector_p),
					('primal', lib.graph_vector_p),
					('primal12', lib.graph_vector_p),
					('dual', lib.graph_vector_p),
					('dual12', lib.graph_vector_p),
					('prev', lib.graph_vector_p),
					('temp', lib.graph_vector_p),
					('m', ct.c_size_t),
					('n', ct.c_size_t)]

	lib.pogs_variables = PogsVariables
	lib.pogs_variables_p = ct.POINTER(lib.pogs_variables)
	lib.POGS_STATE_LENGTH = 6

def attach_pogs_adaptrho_ctypes(lib, single_precision=False):
	include_ok_pogs_datatypes(lib, single_precision=single_precision)

	class AdaptiveRhoParameters(ct.Structure):
		_fields_ = [('delta', lib.ok_float),
					('l', lib.ok_float),
					('u', lib.ok_float),
					('xi', lib.ok_float)]

	lib.adapt_params = AdaptiveRhoParameters
	lib.adapt_params_p = ct.POINTER(lib.adapt_params)

def attach_pogs_ctypes(lib, single_precision=False):
	include_ok_prox(lib, single_precision=single_precision)
	include_ok_pogs_datatypes(lib, single_precision=single_precision)
	include_ok_pogs_adaptrho(lib, single_precision=single_precision)
	include_ok_projector(lib, single_precision=single_precision)
	include_ok_anderson(lib, single_precision=single_precision)

	set_pogs_impl(lib)

	work_fields = []
	flag_fields = []
	priv_fields = []

	if lib.py_pogs_impl == 'dense':
		work_fields = [('A', lib.matrix_p), ('P', lib.direct_projector_p)]
		flag_fields = [
				('m', ct.c_size_t), ('n', ct.c_size_t), ('ord', ct.c_uint)]
		priv_fields = [
				('A_equil', lib.ok_float_p),
				('d', lib.ok_float_p),
				('e', lib.ok_float_p),
				('ATA_cholesky', lib.ok_float_p) ]
		lib.pogs_solver_data_p = lib.ok_float_p
	elif lib.py_pogs_impl == 'abstract':
		include_ok_operator(lib, single_precision=single_precision)
		work_fields = [('A', lib.abstract_operator_p), ('P', lib.projector_p)]
		work_fields += [
				('operator_scale', ct.CFUNCTYPE(
						ct.c_uint, lib.abstract_operator_p, lib.ok_float)),
				('operator_equilibrate', ct.CFUNCTYPE(
						ct.c_uint, ct.c_void_p, lib.abstract_operator_p,
						lib.vector_p, lib.vector_p, lib.ok_float))]
		flag_fields = [('direct', ct.c_int), ('equil_norm', lib.ok_float)]
		priv_fields = [('d', lib.ok_float_p), ('e', lib.ok_float_p)]
		lib.pogs_solver_data_p = lib.abstract_operator_p

	work_fields += [
			('d', lib.vector_p),
			('e', lib.vector_p),
			('normA', lib.ok_float),
			('skinny', ct.c_int),
			('normalized', ct.c_int),
			('equilibrated', ct.c_int),
			('linalg_handle', ct.c_void_p)]

	class PogsWork(ct.Structure):
		_fields_ = work_fields

	lib.pogs_work = PogsWork
	lib.pogs_work_p = ct.POINTER(lib.pogs_work)

	class PogsSolverFlags(ct.Structure):
		_fields_ = flag_fields

	lib.pogs_solver_flags = PogsSolverFlags
	lib.pogs_solver_flags_p = ct.POINTER(lib.pogs_solver_flags)

	class PogsSolverPrivateData(ct.Structure):
		_fields_ = priv_fields

	lib.pogs_solver_priv_data = PogsSolverPrivateData
	lib.pogs_solver_priv_data_p = ct.POINTER(lib.pogs_solver_priv_data)

	class PogsSolver(ct.Structure):
		_fields_ = [('W', lib.pogs_work_p),
					('z', lib.pogs_variables_p),
					('f', lib.function_vector_p),
					('g', lib.function_vector_p),
					('rho', lib.ok_float),
					('settings', lib.pogs_settings_p),
					('linalg_handle', ct.c_void_p),
					('init_time', lib.ok_float),
					('aa', lib.anderson_accelerator_p)]

	lib.pogs_solver = PogsSolver
	lib.pogs_solver_p = ct.POINTER(lib.pogs_solver)

def _attach_pogs_adaptrho_ccalls(lib):
	lib.pogs_adaptive_rho_initialize.argtypes = [lib.adapt_params_p]
	lib.pogs_adapt_rho.argtypes = [
			lib.pogs_variables_p, lib.ok_float_p, lib.adapt_params_p,
			lib.pogs_settings_p, lib.pogs_residuals_p, lib.pogs_tolerances_p,
			ct.c_uint]

	lib.pogs_adaptive_rho_initialize.restype = ct.c_uint
	OptkitLibs.attach_default_restype(lib.pogs_adapt_rho)

def _attach_pogs_common_ccalls(lib):
	# arguments
	lib.pogs_graph_vector_alloc.argtypes = [
			lib.graph_vector_p, ct.c_size_t, ct.c_size_t]
	lib.pogs_graph_vector_free.argtypes = [lib.graph_vector_p]
	lib.pogs_graph_vector_attach_memory.argtypes = [lib.graph_vector_p]
	lib.pogs_graph_vector_release_memory.argtypes = [lib.graph_vector_p]
	lib.pogs_graph_vector_view_vector.argtypes = [
			lib.graph_vector_p, lib.vector_p, ct.c_size_t, ct.c_size_t]

	lib.pogs_variables_alloc.argtypes = [
			lib.pogs_variables_p, ct.c_size_t, ct.c_size_t]
	lib.pogs_variables_free.argtypes = [lib.pogs_variables_p]

	lib.pogs_set_default_settings.argtypes = [lib.pogs_settings_p]
	lib.pogs_update_settings.argtypes = [
			lib.pogs_settings_p, lib.pogs_settings_p]

	lib.pogs_initialize_conditions.argtypes = [
			lib.pogs_objective_values_p, lib.pogs_residuals_p,
			lib.pogs_tolerances_p, lib.pogs_settings_p, ct.c_size_t,
			ct.c_size_t]
	lib.pogs_update_objective_values.argtypes = [
			ct.c_void_p, lib.function_vector_p, lib.function_vector_p,
			lib.ok_float, lib.pogs_variables_p, lib.pogs_objective_values_p]
	lib.pogs_update_tolerances.argtypes = [
			ct.c_void_p, lib.pogs_variables_p, lib.pogs_objective_values_p,
			lib.pogs_tolerances_p]

	lib.pogs_set_print_iter.argtypes = [lib.c_uint_p, lib.pogs_settings_p]
	lib.pogs_print_header_string.argtypes = []
	lib.pogs_print_iter_string.argtypes = [
			lib.pogs_residuals_p, lib.pogs_tolerances_p,
			lib.pogs_objective_values_p, ct.c_uint]

	lib.pogs_scale_objectives.argtypes = [
			lib.function_vector_p, lib.function_vector_p, lib.vector_p,
			lib.vector_p, lib.function_vector_p, lib.function_vector_p]
	lib.pogs_unscale_output.argtypes = [
			lib.pogs_output_p, lib.pogs_variables_p, lib.vector_p,
			lib.vector_p, lib.ok_float, ct.c_uint]

	# results
	OptkitLibs.attach_default_restype(
			lib.pogs_graph_vector_alloc,
			lib.pogs_graph_vector_free,
			lib.pogs_graph_vector_attach_memory,
			lib.pogs_graph_vector_release_memory,
			lib.pogs_graph_vector_view_vector,
			lib.pogs_variables_alloc,
			lib.pogs_variables_free,
			lib.pogs_set_default_settings,
			lib.pogs_update_settings,
			lib.pogs_initialize_conditions,
			lib.pogs_update_objective_values,
			lib.pogs_update_tolerances,
			lib.pogs_set_print_iter,
			lib.pogs_print_header_string,
			lib.pogs_print_iter_string,
			lib.pogs_scale_objectives,
			lib.pogs_unscale_output,
	)

def _attach_pogs_dense_ccalls(lib):
	lib.pogs_dense_problem_data_alloc.argtypes = [
			lib.pogs_work_p, lib.ok_float_p, lib.pogs_solver_flags_p]
	lib.pogs_dense_problem_data_free.argtypes = [lib.pogs_work_p]
	lib.pogs_dense_get_init_data.argtypes = [
			lib.ok_float_p, lib.pogs_solver_priv_data_p,
			lib.pogs_solver_flags_p]

	lib.pogs_dense_apply_matrix.argtypes = [
			lib.pogs_work_p, lib.ok_float, lib.vector_p, lib.ok_float,
			lib.vector_p]
	lib.pogs_dense_apply_adjoint.argtypes = [
			lib.pogs_work_p, lib.ok_float, lib.vector_p, lib.ok_float,
			lib.vector_p]
	lib.pogs_dense_project_graph.argtypes = [
			lib.pogs_work_p, lib.vector_p, lib.vector_p, lib.vector_p,
			lib.vector_p, lib.ok_float]

	lib.pogs_dense_equilibrate_matrix.argtypes = [
			lib.pogs_work_p, lib.ok_float_p, lib.pogs_solver_flags_p]
	lib.pogs_dense_initalize_graph_projector.argtypes = [lib.pogs_work_p]
	lib.pogs_dense_estimate_norm.argtypes = [lib.pogs_work_p, lib.ok_float_p]
	lib.pogs_dense_work_get_norm.argtypes = [lib.pogs_work_p]
	lib.pogs_dense_work_normalize.argtypes = [lib.pogs_work_p]

	lib.pogs_dense_save_work.argtypes = [
			lib.pogs_solver_priv_data_p, lib.pogs_solver_flags_p,
			lib.pogs_work_p]
	lib.pogs_dense_load_work.argtypes = [
			lib.pogs_work_p, lib.pogs_solver_priv_data_p,
			lib.pogs_solver_flags_p]

	OptkitLibs.attach_default_restype(
			lib.pogs_dense_problem_data_alloc,
			lib.pogs_dense_problem_data_free,
			lib.pogs_dense_get_init_data,
			lib.pogs_dense_apply_matrix,
			lib.pogs_dense_apply_adjoint,
			lib.pogs_dense_project_graph,
			lib.pogs_dense_equilibrate_matrix,
			lib.pogs_dense_initalize_graph_projector,
			lib.pogs_dense_estimate_norm,
			lib.pogs_dense_work_get_norm,
			lib.pogs_dense_work_normalize,
			lib.pogs_dense_save_work,
			lib.pogs_dense_load_work,
	)


def _attach_pogs_abstract_ccalls(lib):
	## arguments
	lib.pogs_abstract_problem_data_alloc.argtypes = [
			lib.pogs_work_p, lib.abstract_operator_p, lib.pogs_solver_flags_p]
	lib.pogs_abstract_problem_data_free.argtypes = [lib.pogs_work_p]
	lib.pogs_abstract_get_init_data.argtypes = [
			lib.abstract_operator_p, lib.pogs_solver_priv_data_p,
			lib.pogs_solver_flags_p]

	lib.pogs_abstract_apply_matrix.argtypes = [
			lib.pogs_work_p, lib.ok_float, lib.vector_p, lib.ok_float,
			lib.vector_p]
	lib.pogs_abstract_apply_adjoint.argtypes = [
			lib.pogs_work_p, lib.ok_float, lib.vector_p, lib.ok_float,
			lib.vector_p]
	lib.pogs_abstract_project_graph.argtypes = [
			lib.pogs_work_p, lib.vector_p, lib.vector_p, lib.vector_p,
			lib.vector_p, lib.ok_float]

	lib.pogs_abstract_equilibrate_matrix.argtypes = [
			lib.pogs_work_p, lib.abstract_operator_p,
			lib.pogs_solver_flags_p]
	lib.pogs_abstract_initalize_graph_projector.argtypes = [lib.pogs_work_p]
	lib.pogs_abstract_estimate_norm.argtypes = [lib.pogs_work_p, lib.ok_float_p]
	lib.pogs_abstract_work_get_norm.argtypes = [lib.pogs_work_p]
	lib.pogs_abstract_work_normalize.argtypes = [lib.pogs_work_p]

	lib.pogs_abstract_save_work.argtypes = [
			lib.pogs_solver_priv_data_p, lib.pogs_solver_flags_p,
			lib.pogs_work_p]
	lib.pogs_abstract_load_work.argtypes = [
			lib.pogs_work_p, lib.pogs_solver_priv_data_p,
			lib.pogs_solver_flags_p]

	lib.pogs_dense_operator_gen.argtypes = [
			lib.ok_float_p, ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.pogs_sparse_operator_gen.argtypes = [
			lib.ok_float_p, lib.ok_int_p, lib.ok_int_p, ct.c_size_t,
			ct.c_size_t, ct.c_size_t, ct.c_uint]
	lib.pogs_dense_operator_free.argtypes = [lib.abstract_operator_p]
	lib.pogs_sparse_operator_free.argtypes = [lib.abstract_operator_p]

	## return types
	OptkitLibs.attach_default_restype(
			lib.pogs_abstract_problem_data_alloc,
			lib.pogs_abstract_problem_data_free,
			lib.pogs_abstract_get_init_data,
			lib.pogs_abstract_apply_matrix,
			lib.pogs_abstract_apply_adjoint,
			lib.pogs_abstract_project_graph,
			lib.pogs_abstract_equilibrate_matrix,
			lib.pogs_abstract_initalize_graph_projector,
			lib.pogs_abstract_estimate_norm,
			lib.pogs_abstract_work_get_norm,
			lib.pogs_abstract_work_normalize,
			lib.pogs_abstract_save_work,
			lib.pogs_abstract_load_work,
			lib.pogs_dense_operator_free,
			lib.pogs_sparse_operator_free,
	)
	lib.pogs_dense_operator_gen.restype = lib.abstract_operator_p
	lib.pogs_sparse_operator_gen.restype = lib.abstract_operator_p


def _attach_pogs_generic_ccalls(lib):
	## arguments
	lib.pogs_work_alloc.argtypes = [
			lib.pogs_work_p, lib.pogs_solver_data_p, lib.pogs_solver_flags_p]
	lib.pogs_work_free.argtypes = [lib.pogs_work_p]

	lib.pogs_solver_alloc.argtypes = [
			lib.pogs_solver_p, lib.pogs_solver_data_p, lib.pogs_solver_flags_p]
	lib.pogs_solver_free.argtypes = [lib.pogs_solver_p]

	lib.pogs_normalize_DAE.argtypes = [lib.pogs_work_p]
	lib.pogs_set_z0.argtypes = [lib.pogs_solver_p]

	lib.pogs_primal_update.argtypes = [lib.pogs_variables_p]
	lib.pogs_prox.argtypes = [
			ct.c_void_p, lib.function_vector_p, lib.function_vector_p,
			lib.pogs_variables_p, lib.ok_float]
	lib.pogs_project_graph.argtypes = [
			lib.pogs_work_p, lib.pogs_variables_p, lib.ok_float, lib.ok_float]
	lib.pogs_dual_update.argtypes = [
			ct.c_void_p, lib.pogs_variables_p, lib.ok_float]

	lib.pogs_iterate.argtypes = [lib.pogs_solver_p]
	lib.pogs_accelerate.argtypes = [lib.pogs_solver_p]
	lib.pogs_update_residuals.argtypes = [
			lib.pogs_solver_p, lib.pogs_objective_values_p,
			lib.pogs_residuals_p]
	lib.pogs_check_convergence.argtypes = [
			lib.pogs_solver_p, lib.pogs_objective_values_p,
			lib.pogs_residuals_p, lib.pogs_tolerances_p, lib.c_int_p]

	lib.pogs_setup_diagnostics.argtypes = [lib.pogs_solver_p, ct.c_uint]
	lib.pogs_record_diagnostics.argtypes = [lib.pogs_solver_p,
			lib.pogs_residuals_p, lib.pogs_tolerances_p, ct.c_uint]
	lib.pogs_emit_diagnostics.argtypes = [lib.pogs_output_p, lib.pogs_solver_p]

	lib.pogs_solver_loop.argtypes = [lib.pogs_solver_p, lib.pogs_info_p]

	lib.pogs_init.argtypes = [lib.pogs_solver_data_p, lib.pogs_solver_flags_p]
	lib.pogs_solve.argtypes = [
			lib.pogs_solver_p, lib.function_vector_p, lib.function_vector_p,
			lib.pogs_settings_p, lib.pogs_info_p, lib.pogs_output_p]
	lib.pogs_finish.argtypes = [lib.pogs_solver_p, ct.c_int]
	lib.pogs.argtypes = [
			lib.pogs_solver_data_p, lib.pogs_solver_flags_p,
			lib.function_vector_p, lib.function_vector_p, lib.pogs_settings_p,
			lib.pogs_info_p, lib.pogs_output_p, ct.c_int]

	lib.pogs_export_solver.argtypes = [
			lib.pogs_solver_priv_data_p, lib.ok_float_p, lib.ok_float_p,
			lib.pogs_solver_flags_p, lib.pogs_solver_p]
	lib.pogs_load_solver.argtypes = [
			lib.pogs_solver_priv_data_p, lib.ok_float_p, lib.ok_float,
			lib.pogs_solver_flags_p]

	lib.pogs_solver_save_state.argtypes = [
			lib.ok_float_p, lib.ok_float_p, lib.pogs_solver_p]
	lib.pogs_solver_load_state.argtypes = [
			lib.pogs_solver_p, lib.ok_float_p, lib.ok_float]

	## return types
	OptkitLibs.attach_default_restype(
			lib.pogs_work_alloc,
			lib.pogs_work_free,
			lib.pogs_solver_alloc,
			lib.pogs_solver_free,
			lib.pogs_normalize_DAE,
			lib.pogs_set_z0,
			lib.pogs_primal_update,
			lib.pogs_prox,
			lib.pogs_project_graph,
			lib.pogs_dual_update,
			lib.pogs_iterate,
			lib.pogs_accelerate,
			lib.pogs_update_residuals,
			lib.pogs_check_convergence,
			lib.pogs_setup_diagnostics,
			lib.pogs_record_diagnostics,
			lib.pogs_emit_diagnostics,
			lib.pogs_solver_loop,
			lib.pogs_solve,
			lib.pogs_finish,
			lib.pogs,
			lib.pogs_export_solver,
			lib.pogs_solver_save_state,
			lib.pogs_solver_load_state,
	)

	lib.pogs_init.restype = lib.pogs_solver_p
	lib.pogs_load_solver.restype = lib.pogs_solver_p


def attach_pogs_ccalls(lib, single_precision=False):
	include_ok_pogs(lib, single_precision=single_precision)
	lib.c_uint_p = ct.POINTER(ct.c_uint)

	_attach_pogs_adaptrho_ccalls(lib)
	_attach_pogs_common_ccalls(lib)
	_attach_pogs_generic_ccalls(lib)

	# DENSE_IMPL CALLS
	if lib.py_pogs_impl == 'dense':
		_attach_pogs_dense_ccalls(lib)

	# ABSTRACT_IMPL CALLS
	if lib.py_pogs_impl == 'abstract':
		_attach_pogs_abstract_ccalls(lib)
