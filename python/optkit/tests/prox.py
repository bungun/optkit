from optkit.api import *
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.utils.pyutils import println, printvoid, var_assert
from optkit.types import ok_function_enums
from optkit.tests.defs import TEST_EPS
import numpy as np
from numpy import inf,nan

VEC_ASSERT = lambda *v : var_assert(*v,type=Vector)
FUNC_ASSERT = lambda *f : var_assert(*f,type=FunctionVector)

def test_prox(*args, **kwargs):
	print "FUNCTION AND PROXIMAL OPERATOR TESTING\n\n\n\n"
	m = kwargs['shape'][0] if 'shape' in kwargs else 5
	m = min(m, 5)
	PRINT = println if '--verbose' in args else printvoid
	PRINTFUNC = print_function_vector if '--verbose' in args else printvoid


	for FUNCKEY in ok_function_enums.enum_dict.keys():	
		print "\n", FUNCKEY, "\n"

		f = FunctionVector(m, h=FUNCKEY)
		assert FUNC_ASSERT(f)
		PRINT("INITIALIZE FUNCTION VECTOR WITH h={}".format(FUNCKEY))
		PRINT("(C CONSTRUCTOR INITIALIZES TO h=0)")
		PRINTFUNC(f)

		PRINT("PYTHON VALUES FOR h:")
		PRINT(f.h_)

		PRINT("SYNC FUNCTION VECTOR TO SET h VALUE")
		push_function_vector(f)
		PRINTFUNC(f)

		PRINT("MODIFY FUNCTION VECTOR\nf.b += 0.3; f.c += 2; f.d -= 0.45")
		f.b_ += 0.3
		f.c_ += 2.
		f.d_ -= 0.45
		push_function_vector(f)
		PRINTFUNC(f)

		rho = 1.


		x = Vector(np.random.rand(m))
		x_out = Vector(np.random.rand(m))
		x_orig = np.copy(x.py)
		
		assert VEC_ASSERT(x, x_out)
		PRINT('PROX EVALUATION INPUT:')
		PRINT(x.py)
		PRINT('PROX EVALUATION OUPUT TARGET---TO BE OVERWRITTEN')
		PRINT(x_out.py)

		x_out_py = prox_eval_python(f, rho, x.py, func=FUNCKEY)
		prox_eval(f,rho,x,x_out)
		sync(x, x_out)


		PRINT('PROX EVALUATION C')
		PRINT(x_out.py)
		PRINT('PROX EVALUATION PYTHON')
		PRINT(x_out_py)

		assert np.max(np.abs(x_out.py - x_out_py)) <= TEST_EPS

		#prox eval should not change x value
		assert np.max(np.abs(x.py - x_orig)) == 0.

		copy(x_out, x)
		sync(x)
		x_orig = np.copy(x.py)
		PRINT('FUNCTION EVALUATION INPUT:')
		PRINT(x.py)
		res_c = func_eval(f, x)
		res_py = func_eval_python(f, x.py, func=FUNCKEY)
		sync(x)
		PRINT('FUNCTION EVALUATION C')
		PRINT(res_c)
		PRINT('FUNCTION EVALUATION PYTHON')
		PRINT(res_py)

		assert abs(res_c-res_py) <= TEST_EPS

		#func eval should not change x value
		assert np.max(np.abs(x.py-x_orig)) == 0.


		# modify values
		f.set(start=m-3, a=4.5, b=[1,2,3])
		for i in xrange(3):
			assert f.a_[-(i+1)] == 4.5
			assert f.b_[-(i+1)] == 3-i


		# copy
		f1 = FunctionVector(m)
		f1.copy_from(f)
		assert all(f.a_ - f1.a_ == 0)
		assert all(f.b_ - f1.b_ == 0)
		assert all(f.c_ - f1.c_ == 0)
		assert all(f.d_ - f1.d_ == 0)
		assert all(f.e_ - f1.e_ == 0)
		assert all(f.h_ - f1.h_ == 0)

		# copy initalizer
		f2 = FunctionVector(m, f=f)
		assert all(f.a_ - f1.a_ == 0)
		assert all(f.b_ - f1.b_ == 0)
		assert all(f.c_ - f1.c_ == 0)
		assert all(f.d_ - f1.d_ == 0)
		assert all(f.e_ - f1.e_ == 0)
		assert all(f.h_ - f1.h_ == 0)		


	print "...passed"
	return True