from operator import add as op_add
from numpy.random import rand
from numpy import zeros, float64

def gen_test_defs(backend):
	TEST_EPS=1e-5 if backend.lowtypes.FLOAT_CAST == float64 else 1e-3
	MAT_ORDER = 'F' if backend.layout == 'col' else 'C'

	def RAND_ARR(*dims):
		if len(dims)==2 and MAT_ORDER != 'C':
			arr = zeros(shape=dims, order='F')
			arr[:] = rand(*dims)[:]
			return arr.astype(backend.lowtypes.FLOAT_CAST)
		else:
			return rand(*dims).astype(backend.lowtypes.FLOAT_CAST)


	return TEST_EPS, RAND_ARR, MAT_ORDER