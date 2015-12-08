from optkit import Vector, FunctionVector
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.utils.pyutils import println,printvoid,var_assert
from optkit.types import ok_function_enums
from optkit.kernels import proximal as proxops
from optkit.tests.defs import TEST_EPS

import numpy as np
from numpy import inf,nan

VEC_ASSERT = lambda *v : var_assert(*v,type=Vector)
FUNC_ASSERT = lambda *f : var_assert(*f,type=FunctionVector)





def test_prox(*args, **kwargs):
	print "FUNCTION AND PROXIMAL OPERATOR TESTING\n\n\n\n"
	m = kwargs['shape'][0] if 'shape' in kwargs else 5
	m = min(m, 5)
	PRINT=println if '--verbose' in args else printvoid
	PRINTFUNC=proxops.print_function_vector if '--verbose' in args else printvoid

	assert proxops is not None
	PRINT(proxops)


	for FUNCKEY in ok_function_enums.enum_dict.keys():	
		print "\n", FUNCKEY, "\n"

		f = FunctionVector(m,h=FUNCKEY)
		assert FUNC_ASSERT(f)
		PRINTFUNC(f)
		PRINT(f.h_)
		proxops.push_function_vector(f)
		PRINTFUNC(f)
		f.b_ += 0.3
		f.c_ += 2.
		f.d_ -= 0.45
		proxops.push_function_vector(f)
		PRINTFUNC(f)

		rho = 1.


		x = Vector(np.random.rand(m))
		x_out = Vector(np.random.rand(m))
		x_orig = np.copy(x.py)
		assert VEC_ASSERT(x,x_out)

		PRINT(x.py)
		PRINT(x_out.py)
		proxops.prox(f,rho,x,x_out)
		PRINT(x_out.py)
		x_out_py = prox_eval_python(f,rho,x.py,func=FUNCKEY)
		assert np.max(np.abs(x_out.py-x_out_py)) <= TEST_EPS

		#prox eval should not change x value
		assert np.max(np.abs(x.py-x_orig)) == 0.

		x.py[:]=x_out.py[:]
		x_orig[:]=x.py[:]
		PRINT(x.py)
		res_c = proxops.eval(f,x)
		res_py = func_eval_python(f,x.py,func=FUNCKEY)
		assert abs(res_c-res_py) <= TEST_EPS
		PRINT(res_c)
		#func eval should not change x value
		assert np.max(np.abs(x.py-x_orig)) == 0.


		# modify values
		f.set(start=m-3,a=4.5,b=[1,2,3])
		for i in xrange(3):
			assert f.a_[-(i+1)] == 4.5
			assert f.b_[-(i+1)] == 3-i


	print "...passed"
	return True
