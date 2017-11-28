from optkit.compat import *

import os
import numpy as np
import numpy.linalg as la
import ctypes as ct

import optkit.libs.enums as enums
import optkit.libs.error as okerr

PRINTERR = okerr.optkit_print_error
TEST_ITERATE = int(os.getenv('OPTKIT_REPEAT_NUMERICALTEST', '0'))
VERBOSE_TEST = os.getenv('OPTKIT_TEST_VERBOSE', False)

def noerr(c_call_status):
    return PRINTERR(c_call_status) == 0

def vec_equal(first, second, atol, rtol):
    lhs = la.norm(first - second)
    rhs = atol + rtol * la.norm(second)
    print "LHS", lhs
    print "RHS", rhs
    if not lhs <= rhs:
        print('vector comparison failure:\n'
              '||a - b||: {}\n'
              'atol + rol||b||: {}'
              ''.format(lhs, rhs))
    return lhs <= rhs

def scalar_equal(first, second, tol):
    return abs(first - second) <= tol * (1 + abs(second))

def standard_vector_tolerances(lib, m, modulate_gpu=0):
    rtol = 10**(-7 + 2 * lib.FLOAT + modulate_gpu * lib.GPU)
    atol = rtol * m**0.5
    return rtol, atol

def standard_tolerances(lib, m, n, modulate_gpu=0):
    rtol = 10**(-7 + 2 * lib.FLOAT + modulate_gpu * lib.GPU)
    atolm = rtol * m**0.5
    atoln = rtol * n**0.5
    atolmn = rtol * (m + n)**0.5
    return rtol, atolm, atoln, atolmn