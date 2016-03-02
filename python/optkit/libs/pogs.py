from ctypes import POINTER, Structure, c_int, c_uint, c_size_t, c_void_p
from optkit.libs.loader import retrieve_libs, validate_lib

class PogsLibs(object):
	def __init__(self, dense=True):
		if dense:
			self.libs, search_results = retrieve_libs('libpogs_dense_')
		else:
			self.libs, search_results = retrieve_libs('libpogs_sparse_')
		if all([self.libs[k] is None for k in self.libs]):
			raise ValueError('No backend libraries were located:\n{}'.format(
				search_results))

	def get(self, denselib, proxlib, single_precision=False, gpu=False):
		device = 'gpu' if gpu else 'cpu'
		precision = '32' if single_precision else '64'
		lib_key = '{}{}'.format(device, precision)

		if lib_key not in self.libs:
			return None
		elif self.libs[lib_key] is None:
			return None

		validate_lib(denselib, 'denselib', 'vector_calloc', type(self),
			single_precision, gpu)
		validate_lib(proxlib, 'proxlib', 'function_vector_calloc', type(self),
			single_precision, gpu)

		lib = self.libs[lib_key]
		if lib.INITIALIZED:
			return lib
		else:
			ok_float = denselib.ok_float
			ok_float_p = denselib.ok_float_p
			vector_p = denselib.vector_p
			matrix_p = denselib.matrix_p
			function_vector_p = proxlib.function_vector_p

			# Public API
			# ----------

			## enums
			class OKEquilibrationEnums(object):
				EquilSinkhorn = c_uint(0).value
				EquilDenseL2 = c_uint(1).value

			lib.enums = OKEquilibrationEnums()

			## structs
			class PogsSettings(Structure):
				_fields_ = [('alpha', ok_float),
							('rho', ok_float),
							('abstol', ok_float),
							('reltol', ok_float),
							('maxiter', c_uint),
							('verbose', c_uint),
							('suppress', c_uint),
							('adaptiverho', c_int),
							('gapstop', c_int),
							('warmstart', c_int),
							('resume', c_int),
							('x0', ok_float_p),
							('nu0', ok_float_p)]

			pogs_settings_p = POINTER(PogsSettings)
			lib.pogs_settings = PogsSettings
			lib.pogs_settings_p = pogs_settings_p

			class PogsInfo(Structure):
				_fields_ = [('err', c_int),
							('converged', c_int),
							('k', c_uint),
							('obj', ok_float),
							('rho', ok_float),
							('setup_time', ok_float),
							('solve_time', ok_float)]

			pogs_info_p = POINTER(PogsInfo)
			lib.pogs_info = PogsInfo
			lib.pogs_info_p = pogs_info_p

			class PogsOutput(Structure):
				_fields_ = [('x', ok_float_p),
							('y', ok_float_p),
							('mu', ok_float_p),
							('nu', ok_float_p)]

			pogs_output_p = POINTER(PogsOutput)
			lib.pogs_output = PogsOutput
			lib.pogs_output_p = pogs_output_p


			## arguments
			lib.set_default_settings.argtypes = [pogs_settings_p]
			lib.pogs_init.argtypes = [ok_float_p, c_size_t, c_size_t, c_uint,
									  c_uint]
			lib.pogs_solve.argtypes = [c_void_p, function_vector_p,
									   function_vector_p, pogs_settings_p,
									   pogs_info_p, pogs_output_p]
			lib.pogs_finish.argtypes = [c_void_p, c_int]
			lib.pogs.argtypes = [ok_float_p, function_vector_p,
								 function_vector_p, pogs_settings_p,
								 pogs_info_p, pogs_output_p, c_uint, c_uint]
			lib.pogs_load_solver.argtypes = [ok_float_p, ok_float_p,
											 ok_float_p, ok_float_p,
											 ok_float_p, ok_float_p,
											 ok_float_p, ok_float_p,
											 ok_float_p, ok_float, c_size_t,
											 c_size_t, c_uint]
			lib.pogs_extract_solver.argtypes = [c_void_p, ok_float_p,
												ok_float_p, ok_float_p,
												ok_float_p, ok_float_p,
												ok_float_p, ok_float_p,
												ok_float_p, ok_float_p,
												ok_float_p, c_uint]

			## return types
			lib.set_default_settings.restype = None
			lib.pogs_init.restype = c_void_p
			lib.pogs_solve.restype = None
			lib.pogs_finish.restype = None
			lib.pogs.restype = None
			lib.pogs_load_solver.restype = c_void_p
			lib.pogs_extract_solver.restype = None

			lib.private_api_accessible.restype = c_int
			lib.full_api_accessible = lib.private_api_accessible()

			# function to test whether compiled for
			# direct/indirect projection
			lib.is_direct.restype = c_int
			lib.direct = lib.is_direct()

			# Private API
			# -----------
			if lib.full_api_accessible:

				## structs
				class AdaptiveRhoParameters(Structure):
					_fields_ = [('delta', ok_float),
								('l', ok_float),
								('u', ok_float),
								('xi', ok_float)]

				adapt_params_p = POINTER(AdaptiveRhoParameters)
				lib.adapt_params = AdaptiveRhoParameters
				lib.adapt_params_p = adapt_params_p


				class PogsBlockVector(Structure):
					_fields_ = [('size', c_size_t),
								('m', c_size_t),
								('n', c_size_t),
								('x', vector_p),
								('y', vector_p),
								('vec', vector_p)]


				block_vector_p = POINTER(PogsBlockVector)
				lib.block_vector = PogsBlockVector
				lib.block_vector_p = block_vector_p

				class PogsResiduals(Structure):
					_fields_ = [('primal', ok_float),
								('dual', ok_float),
								('gap', ok_float)]

				pogs_residuals_p = POINTER(PogsResiduals)
				lib.pogs_residuals = PogsResiduals
				lib.pogs_residuals_p = pogs_residuals_p

				class PogsTolerances(Structure):
					_fields_ = [('primal', ok_float),
								('dual', ok_float),
								('gap', ok_float),
								('reltol', ok_float),
								('abstol', ok_float),
								('atolm', ok_float),
								('atoln', ok_float),
								('atolmn', ok_float)]

				pogs_tolerances_p = POINTER(PogsTolerances)
				lib.pogs_tolerances = PogsTolerances
				lib.pogs_tolerances_p = pogs_tolerances_p

				class PogsObjectives(Structure):
					_fields_ = [('primal', ok_float),
								('dual', ok_float),
								('gap', ok_float)]

				pogs_objectives_p = POINTER(PogsObjectives)
				lib.pogs_objectives = PogsObjectives
				lib.pogs_objectives_p = pogs_objectives_p

				class PogsMatrix(Structure):
					_fields_ = [('A', matrix_p),
								('P', c_void_p),
								('d', vector_p),
								('e', vector_p),
								('normA', ok_float),
								('skinny', c_int),
								('normalized', c_int),
								('equilibrated', c_int)]

				pogs_matrix_p = POINTER(PogsMatrix)
				lib.pogs_matrix = PogsMatrix
				lib.pogs_matrix_p = pogs_matrix_p

				class PogsVariables(Structure):
					_fields_ = [('primal', block_vector_p),
								('primal12', block_vector_p),
								('dual', block_vector_p),
								('dual12', block_vector_p),
								('prev', block_vector_p),
								('temp', block_vector_p),
								('m', c_size_t),
								('n', c_size_t)]

				pogs_variables_p = POINTER(PogsVariables)
				lib.pogs_variables = PogsVariables
				lib.pogs_variables_p = pogs_variables_p


				class PogsSolver(Structure):
					_fields_ = [('M', pogs_matrix_p),
								('z', pogs_variables_p),
								('f', function_vector_p),
								('g', function_vector_p),
								('rho', ok_float),
								('settings', pogs_settings_p),
								('linalg_handle', c_void_p),
								('init_time', ok_float)]

				pogs_solver_p = POINTER(PogsSolver)
				lib.pogs_solver = PogsSolver
				lib.pogs_solver_p = pogs_solver_p

				## argtypes
				lib.update_problem.argtypes = [pogs_solver_p,
											   function_vector_p,
											   function_vector_p]
				lib.initialize_variables.argtypes = [pogs_solver_p]
				lib.pogs_solver_loop.argtypes = [pogs_solver_p, pogs_info_p]
				lib.make_tolerances.argtypes = [pogs_settings_p, c_size_t,
												c_size_t]
				lib.set_prev.argtypes = [pogs_variables_p]
				lib.prox.argtypes = [c_void_p, function_vector_p,
									 function_vector_p, pogs_variables_p,
									 ok_float]
				lib.project_primal.argtypes = [c_void_p, c_void_p,
											   pogs_variables_p, ok_float]
				lib.update_dual.argtypes = [c_void_p, pogs_variables_p,
											ok_float]
				lib.check_convergence.argtypes = [c_void_p, pogs_solver_p,
												  pogs_objectives_p,
												  pogs_residuals_p,
												  pogs_tolerances_p]
				lib.adaptrho.argtypes = [pogs_solver_p, adapt_params_p,
					pogs_residuals_p, pogs_tolerances_p, c_uint]
				lib.copy_output.argtypes = [pogs_solver_p, pogs_output_p]

				## results
				lib.update_problem.restype = None
				lib.initialize_variables.restype = None
				lib.pogs_solver_loop.restype = None
				lib.make_tolerances.restype = PogsTolerances
				lib.set_prev.restype = None
				lib.prox.restype = None
				lib.project_primal.restype = None
				lib.update_dual.restype = None
				lib.check_convergence.restype = c_int
				lib.adaptrho.restype = None
				lib.copy_output.restype = None

				if lib.is_direct():
					lib.direct_projector_project.argtypes = [c_void_p,
						c_void_p, vector_p, vector_p, vector_p, vector_p]
					lib.direct_projector_project.restype = None
				else:
					lib.indirect_projector_project.argtypes = [c_void_p,
						c_void_p, vector_p, vector_p, vector_p, vector_p]
					lib.indirect_projector_project.restype = None


				# redefinitions
				lib.pogs_init.restype = pogs_solver_p
				lib.pogs_load_solver.restype = pogs_solver_p

			else:
				# using public API only, define private API fields anyway
				lib.adapt_params = AttributeError
				lib.block_vector = AttributeError
				lib.pogs_residuals = AttributeError
				lib.pogs_tolerances = AttributeError
				lib.pogs_objectives = AttributeError
				lib.pogs_matrix = AttributeError
				lib.pogs_variables = AttributeError
				lib.pogs_solver = AttributeError

				lib.adapt_params_p = AttributeError
				lib.block_vector_p = AttributeError
				lib.pogs_residuals_p = AttributeError
				lib.pogs_tolerances_p = AttributeError
				lib.pogs_objectives_p = AttributeError
				lib.pogs_matrix_p = AttributeError
				lib.pogs_variables_p = AttributeError
				lib.pogs_solver_p = AttributeError

			lib.FLOAT = single_precision
			lib.GPU = gpu
			lib.INITIALIZED = True
			return lib
