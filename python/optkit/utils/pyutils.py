from optkit.types import ok_enums as enums
from ctypes import c_uint
from operator import and_, add as op_add
from toolz import curry 
from numpy import ndarray, squeeze, abs as np_abs, max as np_max

def istypedtuple(x,n, type_):
	valid = isinstance(x,tuple)
	if valid:
		valid &= len(x)==int(n)
		valid &= all(map(lambda y: type(y)==type_, x))
	return valid

def println(*args):
	for arg in args: print arg

def pretty_print(msg, sym='-'):
	line = reduce(op_add, [sym for char in msg])
	print line
	println(msg)
	print line

def printvoid(*args):
	pass

def var_assert(*var, **vartype):
	for v in var:
		if 'type' in vartype:
			assert isinstance(v,vartype['type'])
		else:
			assert v is not None
		if 'selfchecking' in vartype:
			assert vartype['selfchecking']==('isvalid' in dir(v))
		if 'isvalid' in  v.__dict__:
			assert v.isvalid() 
	return True

@curry 
def rel_compare(eps, first, second):
	if first == 0 and second == 0: return True
	return abs(first - second) / max(abs(first), abs(second)) < eps

def array_compare(a1, a2, eps=0., expect=True):
	assert isinstance(a1, ndarray)
	assert isinstance(a2, ndarray)
	assert squeeze(a1).shape == squeeze(a2).shape

	# check absolute tolerance
	valid = np_max(np_abs(a1-a2)) <= eps

	# check relative tolerance
	if not valid and eps > 0:
		rcomp = rel_compare(eps)
		valid |= reduce(and_, map(rcomp, 
			[a1.item(i) for i in xrange(a1.size)],
			[a2.item(i) for i in xrange(a2.size)]))

	if expect and not valid:
		print "ARRAY MISMATCH GREATER THAN TOLERANCE {}:".format(eps)
		print "ARRAY 1\n", a1
		print "ARRAY 2\n", a2

	return valid
