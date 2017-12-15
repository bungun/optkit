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

def tols(rtol, m, n):
    return rtol, rtol * m**0.5, rtol * n**0.5, rtol * (m+n)**0.5

def standard_tolerances(lib, m, n, modulate_gpu=False, repeat_factor=1.):
    return tols(
            float(repeat_factor) *
            10**(-7 + 2 * lib.FLOAT + int(modulate_gpu) * lib.GPU),
            m, n)

def custom_tolerances(lib, m, n, modulate_gpu=0, modulate_float=2, base=7,
                      repeat_factor=1.):
    return tols(
            float(repeat_factor) *
            10**(-base + modulate_float * lib.FLOAT + modulate_gpu * lib.GPU),
            m, n)