import numpy as np
from traceback import format_exc
from numpy import inf,nan
from optkit.utils.proxutils import func_eval_python, prox_eval_python
from optkit.utils.pyutils import println, pretty_print, printvoid, var_assert
from optkit.types import ok_function_enums
from optkit.tests.defs import gen_test_defs

def test_prox(errors, *args, **kwargs):
	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)

		from optkit.api import prox, linsys, Vector, FunctionVector
		TEST_EPS, RAND_ARR, MAT_ORDER = gen_test_defs(backend)	
		VEC_ASSERT = lambda *v : var_assert(*v,type=Vector)
		FUNC_ASSERT = lambda *f : var_assert(*f,type=FunctionVector)


		print "FUNCTION AND PROXIMAL OPERATOR TESTING\n\n\n\n"
		m = kwargs['shape'][0] if 'shape' in kwargs else 5
		m = max(m, 5)
		PRINT = println if '--verbose' in args else printvoid
		PRINTFUNC = prox['print_function_vector'] if '--verbose' in args else printvoid


		for FUNCKEY in ok_function_enums.enum_dict.keys():	
			print "\n", FUNCKEY, "\n"

			f = FunctionVector(m, h=FUNCKEY)
			assert FUNC_ASSERT(f)
			PRINT("INITIALIZE FUNCTION VECTOR WITH h={}".format(FUNCKEY))
			PRINT("(C CONSTRUCTOR INITIALIZES TO h=0)")
			PRINTFUNC(f)

			PRINT("PYTHON VALUES FOR h:")
			PRINT([f.tolist()[i].h for i in xrange(m)])

			PRINT("SYNC FUNCTION VECTOR TO SET h VALUE")
			prox['push_function_vector'](f)
			PRINTFUNC(f)

			PRINT("MODIFY FUNCTION VECTOR\nf.b += 0.3; f.c += 2; f.d -= 0.45")
			f_ = f.tolist()

			for i in xrange(m):
				f_[i].b += 0.3
				f_[i].c += 2.
				f_[i].d -= 0.45 

			f.py[:]=f_[:]
			prox['push_function_vector'](f)
			PRINTFUNC(f)

			rho = 1.

			x = Vector(RAND_ARR(m))
			x_out = Vector(RAND_ARR(m))
			x_orig = np.copy(x.py)
			
			assert VEC_ASSERT(x, x_out)
			PRINT('PROX EVALUATION INPUT:')
			PRINT(x.py)
			PRINT('PROX EVALUATION OUPUT TARGET---TO BE OVERWRITTEN')
			PRINT(x_out.py)

			x_out_py = prox_eval_python(f.tolist(), rho, x.py)
			prox['prox_eval'](f,rho,x,x_out)
			linsys['sync'](x, x_out)

			PRINT('PROX EVALUATION C')
			PRINT(x_out.py)
			PRINT('PROX EVALUATION PYTHON')
			PRINT(x_out_py)

			assert np.max(np.abs(x_out.py - x_out_py)) <= TEST_EPS

			#prox eval should not change x value
			assert np.max(np.abs(x.py - x_orig)) == 0.

			linsys['copy'](x_out, x)
			linsys['sync'](x)
			x_orig = np.copy(x.py)
			PRINT('FUNCTION EVALUATION INPUT:')
			PRINT(x.py)
			res_c = prox['func_eval'](f, x)
			res_py = func_eval_python(f.tolist(), x.py)
			linsys['sync'](x)
			PRINT('FUNCTION EVALUATION C')
			PRINT(res_c)
			PRINT('FUNCTION EVALUATION PYTHON')
			PRINT(res_py)

			assert abs(res_c-res_py) <= TEST_EPS

			#func eval should not change x value
			assert np.max(np.abs(x.py-x_orig)) == 0.


			# modify values
			f.set(start=m-3, a=4.5, b=[1,2,3])
			f.pull()
			fobj = f.tolist()
			for i in xrange(3):
				assert fobj[-(i+1)].a == 4.5
				assert fobj[-(i+1)].b == 3-i


			# copy
			f1 = FunctionVector(m)
			f1.copy_from(f)

			f1.pull()
			fobj1 = f1.tolist()

			assert all([fobj[i].a - fobj1[i].a == 0 for i in xrange(m)])
			assert all([fobj[i].b - fobj1[i].b == 0 for i in xrange(m)])
			assert all([fobj[i].c - fobj1[i].c == 0 for i in xrange(m)])
			assert all([fobj[i].d - fobj1[i].d == 0 for i in xrange(m)])
			assert all([fobj[i].e - fobj1[i].e == 0 for i in xrange(m)])
			assert all([fobj[i].h - fobj1[i].h == 0 for i in xrange(m)])


			# copy initalizer
			f2 = FunctionVector(m, f=f)
			f2.pull()
			fobj2 = f2.tolist()

			assert all([fobj[i].a - fobj2[i].a == 0 for i in xrange(m)])
			assert all([fobj[i].b - fobj2[i].b == 0 for i in xrange(m)])
			assert all([fobj[i].c - fobj2[i].c == 0 for i in xrange(m)])
			assert all([fobj[i].d - fobj2[i].d == 0 for i in xrange(m)])
			assert all([fobj[i].e - fobj2[i].e == 0 for i in xrange(m)])
			assert all([fobj[i].h - fobj2[i].h == 0 for i in xrange(m)])	


		print "...passed"
		return True
	except:
		errors.append(format_exc())
		print "...failed"
		return False