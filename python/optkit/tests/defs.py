from os import getenv, path

VERBOSE_TEST = getenv('OPTKIT_TESTING_VERBOSE', False)

# library conditions: gpu = True/False, single_precision = True/False
CONDITIONS = [(a, b) for a in (True, False) for b in (True, False)]

# todo: modify test code to use env specified test conditions if availabe
DEFAULT_ROWS = getenv('OPTKIT_TESTING_DEFAULT_NROWS', None)
DEFAULT_COLS = getenv('OPTKIT_TESTING_DEFAULT_NCOLS', None)
DEFAULT_MATRIX_PATH = getenv('OPTKIT_TESTING_DEFAULT_MATRIX', None)

if DEFAULT_MATRIX_PATH is not None:
	if not path.exsists(DEFAULT_MATRIX_PATH):
		DEFAULT_MATRIX_PATH = None
	elif not DEFAULT_MATRIX_PATH.endswith('.npy'):
		DEFAULT_MATRIX_PATH = None

def version_string(major, minor, change, status):
	v = "{}.{}.{}".format(major, minor, change)
	if status:
		v.join("-{}".format(chr(status)))
	return v

