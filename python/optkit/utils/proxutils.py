from optkit.compat import *

import numpy as np
# """
# low-level utilities
# """
# class UtilMakeCFunctionVector(object):
# 	def __init__(self, lowtypes, proxlib):
# 		self.proxlib = proxlib
# 	def __call__(self, n=None):
# 		if n is None:
# 			return self.proxlib.function_vector(0, None)
# 		elif isinstance(n, int):
# 			f_ = self.proxlib.function_vector(0,None)
# 			self.proxlib.function_vector_calloc(f_, n)
# 			return f_
# 		else:
# 			return None
# 			# TODO: error message (type, dims)

# class UtilReleaseCFunctionVector(object):
# 	def __init__(self, proxlib):
# 		self.proxlib = proxlib
# 	def __call__(self, f):
# 		if isinstance(f, self.proxlib.function_vector):
# 			self.proxlib.function_vector_free(f)


"""
Python function/prox eval
"""
def lambertw_exp(x):
	"""
	evaluate lambertW(exp(x))

	ref: http://keithbriggs.info/software/LambertW.c
	"""

	if x > 100:
		# approximation for x in [100, 700]
		logx = np.log(x)
		return -0.36962844 + x - 0.97284858 * logx + 1.3437973 / logx
	elif x < 0:
		p = (2 * np.exp(x + 1) + 1)**0.5
		w = -1 + p * (1 + p *(-1. / 3 + p * 11. / 72))
	else:
		w = x

	if x > 1.098612288668110:
		w = -np.log(w)

	for i in xrange(10):
		e = np.exp(w)
		t = w * e - np.exp(x)
		p = w + 1
		t /= e * p - 0.5 * (p + 1) * t / p
		w -= t

	return w

def cubicsolve(p, q, r):
	"""
	Find the root of a cubic x^3 + px^2 + qx + r = 0 with a single positive root
	ref: http://math.stackexchange.com/questions/60376
	"""
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
		B = np.arccos(-b/A)
		C = pow(A, 1/3.)
		return -s + (C - a/C) * np.cos(B/3)

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
		t = 1./(1 + np.exp(-x))
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
		g = 1./(rhoi*(1 + np.exp(-x))) + (x - xi)
		if g > 0:
			l = max(l, x-g)
			u = x
		else:
			u = min(u, x-g)
			l = x
		x = (u + l)/2.
		i += 1

	return x

def enum_to_func(e):
	if e == 0: return 'Zero'
	if e == 1: return 'Abs'
	if e == 2: return 'Exp'
	if e == 3: return 'Huber'
	if e == 4: return 'Identity'
	if e == 5: return 'IndBox01'
	if e == 6: return 'IndEq0'
	if e == 7: return 'IndGe0'
	if e == 8: return 'IndLe0'
	if e == 9: return 'Logistic'
	if e == 10: return 'MaxNeg0'
	if e == 11: return 'MaxPos0'
	if e == 12: return 'NegEntr'
	if e == 13: return 'NegLog'
	if e == 14: return 'Recipr'
	if e == 15: return 'Square'
	if e == 16: return 'Berhu'
	if e == 17: return 'AsymmHuber'
	if e == 18: return 'AsymmSquare'
	if e == 19: return 'AsymmBerhu'
	if e == 20: return 'AffQuad'
	# if e == 21: return 'AffExp'

def ffunc(h, xi):
	func = enum_to_func(h).replace('Asymm', '')
	if func == 'Abs':
		return abs(xi)
	elif func == 'NegEntr':
		return 0 if xi <= 0 else xi * np.log(xi)
	elif func =='Exp':
		return np.exp(xi)
	elif func == 'Huber':
		return abs(xi) - 0.5 if xi >=1 else 0.5 * xi * xi
	elif func == 'Identity':
		return xi
	elif func in ('IndBox01','IndEq0','IndGe0','IndLe0','Zero'):
		return 0
	elif func == 'Logistic':
		return np.log(1 + np.exp(xi))
	elif func == 'MaxNeg0':
		return -xi * (xi < 0)
	elif func == 'MaxPos0':
		return xi * (xi > 0)
	elif func == 'NegLog':
		return -1 * np.log(xi)
	elif func == 'Recipr':
		return (xi**-1) * (xi >= 0)
	elif func == 'Square':
		return 0.5 * xi**2
	elif func == 'Berhu':
		return abs(xi) if xi < 1 else (xi**2 + 1) / 2
	elif func == 'AffQuad':
		return -xi if xi < 0 else 0.5 * xi * xi
	elif func == 'AffExp':
		raise NotImplementedError
	else:
		return 0
	return h

def func_eval_python(f, x):
	"""
	Assumes f is a list of FunctionObject instances, where each element has
	fields a, b, c, d and e of type float32/64 and a field h of type uint
	"""
	val = 0
	for i, ff in enumerate(f):
		xi = ff.a * x[i] - ff.b
		c = ff.c
		funcname = enum_to_func(ff.h)
		if 'Asymm' in funcname or funcname in ('AffQuad', 'AffExp'):
			c *= 1. if xi < 0 else ff.s
		xi = ffunc(ff.h, xi)
		val += c * xi + ff.d * x[i] + 0.5 * ff.e * x[i] * x[i]
	return val

def pfunc(h, xi, rhoi):
	func = enum_to_func(h).replace('Asymm', '')
	if func =='Abs':
		return max(xi - 1./rhoi, 0) + min(xi + 1./rhoi, 0)
	elif func == 'NegEntr':
		return (lambertw_exp(rhoi*xi - 1) * np.log(rhoi)) / rhoi
	elif func == 'Exp':
		return xi - lambertw_exp(xi - np.log(rhoi))
	elif func == 'Huber':
		return xi * rhoi / (1.+rhoi) if abs(xi) < (1 + 1./rhoi) \
			else xi - np.sign(xi) / rhoi
	elif func == 'Identity':
		return xi - 1./rhoi
	elif func == 'IndBox01':
		return min(max(xi, 0), 1)
	elif func == 'IndEq0':
		return 0
	elif func == 'IndGe0':
		return max(xi, 0)
	elif func == 'IndLe0':
		return min(xi, 0)
	elif func == 'Logistic':
		fprox = proxlog(xi, rhoi)
	elif func == 'MaxNeg0':
		return xi + 1./rhoi if xi <= -1./rhoi else max(xi, 0)
	elif func == 'MaxPos0':
		return xi - 1./rhoi if xi >= 1./rhoi else min(xi ,0)
	elif func == 'NegLog':
		return (xi + (xi**2 + 4/rhoi)**0.5) / 2
	elif func == 'Recipr':
		return cubicsolve(-max(xi, 0), 0, -1./rhoi)
	elif func == 'Square':
		return rhoi * xi / (1. + rhoi)
	elif func == 'Berhu':
		if abs(xi) > 1. + 1. / rhoi:
			return rhoi * xi / (1. + rhoi)
		elif xi > 1. / rhoi:
			return xi - 1. / rhoi
		elif xi > -1 / rhoi:
			return 0.
		else:
			return xi + 1 / rhoi
	elif func == 'AffQuad':
		return xi - 1. / rhoi if xi <= - 1. / rhoi else max(
				rhoi * xi / (1. + rhoi), 0)
	elif func == 'AffExp':
		raise NotImplementedError
	else:
		return xi
	return fprox

def prox_eval_python(f, rho, x):
	"""
	Assumes f is a list of FunctionObject instances, where each element has
	fields a, b, c, d and e of type float32/64 and a field h of type uint
	"""
	def fprox(f_, x_):
		x_ = float(x_)
		x_ = f_.a * (x_ * rho - f_.d) / (f_.e + rho) - f_.b
		rho_ = (f_.e + rho) / (f_.c * f_.a * f_.a)
		funcname = enum_to_func(f_.h)
		if 'Asymm' in funcname or funcname in ('AffQuad', 'AffExp'):
			rho_ *= 1. if x_ < 0 else  1. / f_.s
		x_ = pfunc(f_.h, x_, rho_)
		return (x_ + f_.b) / f_.a

	x_out = np.zeros(len(f))
	x_out[:] = listmap(fprox, f, x)

	return x_out