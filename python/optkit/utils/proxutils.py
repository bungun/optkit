from optkit.types import ok_function_enums as fcn_enums
from numpy import log, exp, cos, arccos, sign, inf, nan

"""
low-level utilities
"""
class UtilMakeCFunctionVector(object):
	def __init__(self, lowtypes, proxlib):
		self.lowtypes = lowtypes
		self.proxlib = proxlib
	def __call__(self, n=None):
		if n is None:
			return self.lowtypes.function_vector(0,None)
		elif isinstance(n, int):
			f_ = self.lowtypes.function_vector(0,None)
			self.proxlib.function_vector_calloc(f_, n)
			return f_
		else:
			return None
			# TODO: error message (type, dims)		

class UtilReleaseCFunctionVector(object):
	def __init__(self, lowtypes, proxlib):
		self.lowtypes = lowtypes
		self.proxlib = proxlib
	def __call__(self, f):
		if isinstance(f, self.lowtypes.function_vector):
			self.proxlib.function_vector_free(f)


"""
Python function/prox eval
"""
# ref: http://keithbriggs.info/software/LambertW.c
def lambertw(x):
	EM1 = 0.3678794411714423215955237701614608
  	E = 2.7182818284590452353602874713526625
  	if x == inf or x == nan or x < -EM1:
  		raise ValueError("bad argument ({}) to lambertw.\n"
  						"dom[lambert w]=[-{},infty)".format(
  						x,EM1))
  	elif x == 0: 
  		return 0
  	elif x < -EM1 + 1e-4:
  		q = x + EM1
		w = -1 \
		+2.331643981597124203363536062168 * pow(q,0.5) 	\
		-1.812187885639363490240191647568 * q 			\
		+1.936631114492359755363277457668 * pow(q,1.5) 	\
		-2.353551201881614516821543561516 * pow(q,2)	\
		+3.066858901050631912893148922704 * pow(q,2.5)	\
		-4.175335600258177138854984177460 * pow(q,3)	\
		+5.858023729874774148815053846119 * pow(q,3.5)	\
		-8.401032217523977370984161688514 * pow(q,4)
		return w
  	else:
  		if x < 1:
  			p = pow(2*(E*x+1),0.5)
  			w = (-1+p*(1+p*(-1./3.+p*11./72.)))
  		else:
  			w = log(x)
	  	if x > 3:
	  		w = -log(w) 
	  	for i in xrange(10):
	  		e=exp(w)	
	  		t = w*e-x
	  		p = w+1
	  		t /= (e*p-0.5*(p+1)*(t/p))
	  		w -= t
	  	return w


# /* Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root.
# ref: http://math.stackexchange.com/questions/60376 */
def cubicsolve(p,q,r):
	s = p/3.
	a = -pow(s, 2) + q/3.
	b = pow(s, 3) - s * q/2. + r/2.
	a3 = pow(a, 3)
	b2 = pow(b, 2)
	if a3 + b2 >= 0:
		A = pow(pow(a3 + b2, 0.5) - b, 1/3.)
		return -s - a/A + A
	else:
		A = pow(-a3, 0.5)
		B = arccos(-b/A)
		C = pow(A, 1/3.)
		return -s + (C - a/C) * cos(B/3)

def proxlog(xi,rhoi):
	# initial guess based on piecewise approximation
	if xi < -2.5:
		x = xi
	elif xi > 2.5 + 1./rhoi:
		x = xi - 1./rhoi
	else:
		x = (rhoi * xi - 0.5)/(0.2 + rhoi)

	# Newton iteration
	l = xi - 1./rhoi
	u = xi
	for i in xrange(5):
		t = 1./(1 + exp(-x))
		f = t + rhoi * (x-xi)
		g = t - t**2 + rhoi
		if (f < 0):
			l = x
		else:
			u = x
		x -= f/g
		x = max(min(x, u), l)


	# guarded method if not converged
	i=0
	while i < 100 and u-l > 1e-10:
		g = 1./(rhoi*(1 + exp(-x))) + (x - xi)
		if g > 0:
			l = max(l, x-g)
			u = x
		else:
			u = min(u, x-g)
			l = x
		x = (u + l)/2.
		i += 1

	return x



def func_eval_python(f,x,func='Zero'):
	if func=='Abs':
		h = abs
	elif func=='NegEntr':
		h = lambda x : map(lambda xi: 0 if xi <= 0 else xi*log(xi), x)
	elif func=='Exp':
		h = exp
	elif func=='Huber':
		h = lambda x : (abs(x)-0.5)*(x>=1)+0.5*pow(x,2)*(x<1)
	elif func=='Identity':
		h = lambda x : x 
	elif func in ('IndBox01','IndEq0','IndGe0','IndLe0','Zero'):
		h = lambda x : 0*x
	elif func=='Logistic':
		h = lambda x: log(1+exp(x))
	elif func=='MaxNeg0':
		h = lambda x : -x*(x<0)
	elif func=='MaxPos0':
		h = lambda x :  x*(x>0)
	elif func=='NegLog':
		h = lambda x : -1*log(x)
	elif func=='Recipr':
		h = lambda x : pow(x,-1)*(x>=0)
	elif func=='Square':
		h = lambda x : 0.5*pow(x,2)
	else:
		h = lambda x : 0*x

	return f.c_.dot(h(f.a_*x-f.b_))+f.d_.dot(x)+0.5*f.e_.dot(x*x)


def prox_eval_python(f,rho,x,func='Square'):
	x_out = f.a_*(rho*x-f.d_)
	x_out /= (rho+f.e_)
	x_out -= f.b_

	if func=='Abs':
		fprox = lambda xi, rhoi : max(xi-1./rhoi,0)+min(xi+1./rhoi,0)
	elif func=='NegEntr':
		fprox = lambda xi, rhoi : lambertw(exp(rhoi*xi-1)*rhoi)/rhoi
	elif func=='Exp':
		fprox = lambda xi, rhoi : xi-lambertw(exp(xi)/rhoi)
	elif func=='Huber':
		fprox = lambda xi, rhoi : xi*rhoi/(1.+rhoi) if abs(xi)<(1+1./rhoi) \
									else xi-sign(xi)/rhoi
	elif func=='Identity':
		fprox = lambda xi, rhoi : xi - 1./rhoi
	elif func=='IndBox01':
		fprox = lambda xi, rhoi : min(max(xi,0),1)
	elif func=='IndEq0':
		fprox = lambda xi, rhoi : 0
	elif func=='IndGe0':
		fprox = lambda xi, rhoi : max(xi,0)
	elif func=='IndLe0':
		fprox = lambda xi, rhoi : min(xi,0)
	elif func=='Logistic':
		fprox = proxlog
	elif func=='MaxNeg0':
		fprox = lambda xi, rhoi : xi+1./rhoi if xi <= -1./rhoi else max(xi,0)
	elif func=='MaxPos0':
		fprox = lambda xi, rhoi : xi-1./rhoi if xi >= 1./rhoi else min(xi,0)
	elif func=='NegLog':
		fprox = lambda xi, rhoi : (xi + (xi**2 + 4/rhoi)**0.5)/2
	elif func=='Recipr':
		fprox = lambda xi, rhoi : cubicsolve(-max(xi,0),0,-1./rhoi)
	elif func=='Square':
		fprox = lambda xi, rhoi : rhoi * xi/(1.+rhoi)
	elif func=='Zero':
		fprox = lambda xi, rhoi : xi		
	else:
		fprox = lambda xi, rhoi : xi

	x_out[:] = map(fprox, x_out[:], f.e_ + rho/(f.c_*f.a_**2))

	x_out += f.b_
	x_out /= f.a_

	return x_out