from optkit.compat import *

import operator as op
import numpy as np
import ctypes as ct

def version_string(major, minor, change, status):
    v = "{}.{}.{}".format(major, minor, change)
    if status:
        v+= "-{}".format(chr(status))
    return v

def istypedtuple(x,n, type_):
    valid = isinstance(x,tuple)
    if valid:
        valid &= len(x)==int(n)
        valid &= all(map(lambda y: type(y)==type_, x))
    return valid

def println(*args):
    for arg in args: print(arg)

def pretty_print(msg, sym='-'):
    line = reduce(op.add, [sym for char in msg])
    print(line)
    println(msg)
    print(line)

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

def const_iterator(value, iters):
    for i in xrange(iters):
        yield value