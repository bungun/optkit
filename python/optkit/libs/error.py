from optkit.compat import *

def optkit_print_error(err, print_success=False):
	if err is None:
		return
	elif err == 0:
		if print_success:
			print('\nOPTKIT SUCCESS')
	elif err == 1:
		print('\nOPTKIT ERROR')
	elif err == 2:
		print('\nOPTKIT ERROR CUDA')
	elif err == 3:
		print('\nOPTKIT ERROR CUBLAS')
	elif err == 4:
		print('\nOPTKIT ERROR CUSPARSE')
	elif err == 10:
		print('\nOPTKIT ERROR DOMAIN')
	elif err == 11:
		print('\nOPTKIT ERROR DIVIDE BY ZERO')
	elif err == 100:
		print('\nOPTKIT ERROR LAYOUT MISMATCH')
	elif err == 101:
		print('\nOPTKIT ERROR DIMENSION MISMATCH')
	elif err == 102:
		print('\nOPTKIT ERROR OUT OF BOUNDS')
	elif err == 1000:
		print('\nOPTKIT ERROR OVERWRITE')
	elif err == 1001:
		print('\nOPTKIT ERROR UNALLOCATED')
	else:
		print('\nunrecognized error code: {}'.format(err))

	return err