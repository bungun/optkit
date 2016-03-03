from optkit.backends import OKBackend
from optkit.types import PogsTypes
from os import getenv

"""
Version query
"""
OPTKIT_VERSION = None

"""
Backend handle
"""
backend = OKBackend()

"""
C implementations
"""
pogs_types = None
PogsSolver = None
PogsObjective = None


"""
Backend switching
"""
def set_backend(gpu=False, double=True):

	# Backend
	global OPTKIT_VERSION
	global backend

	## C implementations
	global pogs_types
	global PogsSolver
	global PogsObjective

	# change backend
	backend_name=backend.change(gpu=gpu, double=double)

	OPTKIT_VERSION = backend.version

	## C implemenetations
	pogs_types = PogsTypes(backend)
	PogsSolver = pogs_types.Solver
	PogsObjective = pogs_types.Objective

	print "optkit backend set to {}".format(backend.libname)


"""
INITIALIZATION BEHAVIOR:
"""

default_device = getenv('OPTKIT_DEFAULT_DEVICE', 'cpu')
default_precision = getenv('OPTKIT_DEFAULT_FLOATBITS', '64')


set_backend(gpu=(default_device == 'gpu'),
			double=(default_precision == '64'))