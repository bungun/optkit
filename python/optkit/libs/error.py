def optkit_print_error(err, print_success=False):
	if err is None:
		return
	elif err == 0L:
		if print_success:
			print '\nOPTKIT SUCCESS'
	elif err == 2L:
		print '\nOPTKIT ERROR'
	elif err == 2L:
		print '\nOPTKIT ERROR CUDA'
	elif err == 3L:
		print '\nOPTKIT ERROR CUBLAS'
	elif err == 4L:
		print '\nOPTKIT ERROR CUSPARSE'
	elif err == 10L:
		print '\nOPTKIT ERROR LAYOUT MISMATCH'
	elif err == 11L:
		print '\nOPTKIT ERROR DIMENSION MISMATCH'
	elif err == 12L:
		print '\nOPTKIT ERROR OUT OF BOUNDS'
	elif err == 100L:
		print '\nOPTKIT ERROR OVERWRITE'
	elif err == 101L:
		print '\nOPTKIT ERROR UNALLOCATED'
	else:
		print '\nunrecognized error code: {}'.format(err)

	return err