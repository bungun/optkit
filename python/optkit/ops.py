from toolz import *

def add(x,y, local=False, blas=True):
	#cases: 
	# x, y Vector
	# x, y Matrix
	# x Vector, y scalar (and v-v)
	# x Matrix, y scalar (and v-v)
	pass

def sub(x,y, local=False, blas=True):

	pass

def mul(x,y, local=False, blas=True):
	pass

def div(x,y, local=False):
	pass

def copy(x, y, local=False):
	pass

def cholesky_factor(A, local=False):
	pass

def cholesky_solve(A, x, local=False):
	pass

def sync_vars(x..., local_to_remote=False):
	# cases:
	pass

