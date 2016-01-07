import sys
from optkit.tests import *
from optkit.api import backend

def main(*args, **kwargs):
	tests = []
	passing = 0
	tested = 0
	configs = []
	errors = []
	if '--linsys' in args: tests.append(test_linsys)
	if '--prox' in args: tests.append(test_prox)
	if '--proj' in args: tests.append(test_projector)
	if '--equil' in args: tests.append(test_equil)
	if '--norm' in args: tests.append(test_normalizedprojector)
	if '--block' in args: tests.append(test_blocksplitting)
	if '--pogs' in args: tests.append(test_pogs)
	if '--cequil' in args: tests.append(test_cequil)
	if '--cproj' in args: tests.append(test_cproj)
	if '--cpogs' in args: tests.append(test_cpogs)
	if '--cstore' in args: tests.append(test_cstore)


	if len(tests)==0:
		test_names = ['--linsys', '--prox', '--proj',
		'--equil', '--norm', '--block', '--pogs', '--py_all',
		'--cequil', '--cproj', '--cpogs', '--cstore', '--c_all']

		print str("no tests specified.\nuse optional arguments:\n"
			"{}\nor\n--all\n to specify tests.".format(test_names))
		
	else:
		libkeys = backend.dense_lib_loader.libs

		if libkeys['cpu64'] is not None:
			backend.reset()
			print "<<< CPU, FLOAT64 >>>"
			for t in tests: passing += t(errors, *args, **kwargs)
			tested += len(tests)
			configs.append('cpu64')
		if libkeys['cpu32'] is not None:
			backend.reset()
			print "<<< CPU, FLOAT32 >>>"
			for t in tests: passing += t(errors, 'float', *args, **kwargs)
			tested += len(tests)
			configs.append('cpu32')
		if libkeys['gpu64'] is not None:
			backend.reset()
			print "<<< GPU, FLOAT64 >>>"
			for t in tests: passing += t(errors, 'gpu', *args, **kwargs)
			tested += len(tests)
			configs.append('gpu64')
		if libkeys['gpu32'] is not None:
			backend.reset()
			print "<<< GPU, FLOAT32 >>>"
			for t in tests: passing += t(errors, 'gpu', 'float', *args, **kwargs)
			tested += len(tests)
			configs.append('gpu32')


		print "{}/{} tests passed".format(passing, tested)
		print "configurations tested: ", configs
		if len(errors) > 0:
			print "------------------------------------"
			print "error log"
			for e in errors:
				print "------------------------------------"
				print e
			print "------------------------------------"


if __name__== "__main__":
	args=[]
	kwargs={}

	args += sys.argv
	if '--all' in args:
		args += ['--py_all', '--c_all']
	if '--py_all' in args: 
		args+=['--linsys','--allsub','--prox','--proj','--equil',
			'--norm','--block','--pogs']
	if '--c_all' in args:
		args+=['--cequil', '--cproj', '--cpogs', '--cstore']


	if '--size' in sys.argv:
		pos = sys.argv.index('--size')
		if len(sys.argv) > pos + 2:
			kwargs['shape']=(int(sys.argv[pos+1]),int(sys.argv[pos+2]))
	if '--file' in sys.argv:
		pos = sys.argv.index('--file')
		if len(sys.argv) > pos + 1:
			kwargs['file']=str(sys.argv[pos+1])

	main(*args, **kwargs)