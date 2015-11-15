def istypedtuple(x,n, type_):
	valid = isinstance(x,tuple)
	if valid:
		valid &= len(x)==int(n)
		valid &= all(map(lambda y: type(y)==type_, x))
	return valid
