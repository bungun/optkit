from traceback import format_exc
from ctypes import c_int, byref
from optkit.utils.pyutils import pretty_print 

def version_string(major, minor, change, status):
	v = "{}.{}.{}".format(major, minor, change)
	if status:
		v+= "-{}".format(chr(status))
	return v 

def main(errors, gpu=False, floatbits=64):

	try:
		from optkit.api import backend, set_backend
		if not backend.__LIBGUARD_ON__:
			set_backend(GPU=gpu, double=floatbits == 64)


		major = c_int()
		minor = c_int()
		change = c_int()
		status = c_int()


		backend.dense.denselib_version(byref(major), byref(minor),
			byref(change), byref(status))
		version = version_string(major.value, minor.value,
			change.value, status.value)
		print "DENSE LIB VERSION", version

		backend.sparse.sparselib_version(byref(major), byref(minor),
			byref(change), byref(status))
		version = version_string(major.value, minor.value,
			change.value, status.value)
		print "SPARSE LIB VERSION", version

		backend.prox.proxlib_version(byref(major), byref(minor),
			byref(change), byref(status))
		version = version_string(major.value, minor.value,
			change.value, status.value)
		print "PROX LIB VERSION", version

		backend.pogs.pogslib_version(byref(major), byref(minor),
			byref(change), byref(status))
		version = version_string(major.value, minor.value,
			change.value, status.value)
		print "POGS LIB VERSION", version

		return True

	except:
		errors.append(format_exc())
		return False


def test_version(errors, *args,**kwargs):
	print("\n\n")
	pretty_print("VERSION NUMBER TESTING ...", '#')
	print("\n\n")

	args = list(args)
	floatbits = 32 if 'float' in args else 64
	success = main(errors, gpu='gpu' in args, floatbits=floatbits)

	if success:
		print("\n\n")
		pretty_print("... passed", '#')
		print("\n\n")
	else:
		print("\n\n")
		pretty_print("... failed", '#')
		print("\n\n")
	return success