from optkit.api import backend
from operator import add as op_add
from numpy.random import rand
from numpy import zeros, float64

TEST_EPS=1e-5 if backend.lowtypes.FLOAT_CAST == float64 else 1e-3
HLINE  = reduce(op_add, ['-' for i in xrange(100)]) + "\n"

MAT_ORDER = 'F' if backend.layout == 'col' else 'C'

def rand_arr(*dims):
	if len(dims)==2 and MAT_ORDER != 'C':
		arr = zeros(shape=dims, order='F')
		arr[:] = rand(*dims)[:]
		return backend.lowtypes.FLOAT_CAST(arr)
	else:
		return backend.lowtypes.FLOAT_CAST(rand(*dims))

