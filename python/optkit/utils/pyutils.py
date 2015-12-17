from optkit.types import ok_enums as enums
from ctypes import c_uint

def istypedtuple(x,n, type_):
	valid = isinstance(x,tuple)
	if valid:
		valid &= len(x)==int(n)
		valid &= all(map(lambda y: type(y)==type_, x))
	return valid

def println(*args):
	for arg in args: print arg

def printvoid(*args):
	pass

def var_assert(*var,**vartype):
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