from optkit.compat import *

import numpy.linalg as la

from optkit.tests import defs

def vec_equal(first, second, atol, rtol):
    lhs = la.norm(first - second)
    rhs = atol + rtol * la.norm(second)
    if not lhs <= rhs:
        print('vector comparison failure:\n'
              '||a - b||: {}\n'
              'atol + rtol||b||: {}'
              ''.format(lhs, rhs))
    return lhs <= rhs

def scalar_equal(first, second, tol):
    lhs = abs(first - second)
    rhs = tol * (1 + abs(second))
    if not lhs <= rhs:
        print('scalar comparison failure:\n'
              '|a - b|: {}\n'
              'tol * (1 + |b|): {}'
              ''.format(lhs, rhs))
    return lhs <= rhs

def relaxed_scalar_equal(first, second, tol_nominal, tol_max):
    tol = tol_nominal
    while tol <= tol_max:
        if scalar_equal(first, second, tol):
            return True
        if defs.VERBOSE_TEST:
            print('relaxing tolerance...')
            print('tol curr:', tol)
            print('tol max:', tol_max)
        tol = max(tol_max, tol * 2.)
    return False

def relaxed_vec_equal(first, second, atol_nominal, rtol_nominal, rtol_max):
    atol = atol_nominal
    rtol = rtol_nominal
    while rtol <= rtol_max:
        if vec_equal(first, second, atol, rtol):
            return True
        if defs.VERBOSE_TEST:
            print('relaxing tolerances...')
        atol *= 2
        rtol = max(tol_max, tol * 2.)
    return False


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