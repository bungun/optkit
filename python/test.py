import sys
from optkit.tests import *



def main(*args, **kwargs):
	tests = []
	passing = 0
	if '--linsys' in args: tests.append(test_linsys)
	if '--prox' in args: tests.append(test_prox)
	if '--proj' in args: tests.append(test_projector)
	if '--equil' in args: tests.append(test_equil)
	if '--norm' in args: tests.append(test_normalizedprojector)
	if '--block' in args: tests.append(test_blocksplitting)
	if '--pogs' in args: tests.append(test_pogs)
	for t in tests: passing += t(*args, **kwargs)
	print "{}/{} tests passed".format(passing, len(tests))
	if len(tests)==0:
		print str("no tests specified:\nuse optional arguments:\n"
			"--linsys,\n--prox,\n--proj,\n--equil,\n"
			"--norm,\n--block,\n\n--pogs,\nor\n--all\n to specify tests.")


if __name__== "__main__":
	args=[]
	kwargs={}
	reps=1

	args += sys.argv
	if '--all' in sys.argv: 
		args+=['--linsys','--allsub','--prox','--proj','--equil',
			'--norm','--block','--pogs']

	if '--size' in sys.argv:
		pos = sys.argv.index('--size')
		if len(sys.argv) > pos + 2:
			kwargs['shape']=(int(sys.argv[pos+1]),int(sys.argv[pos+2]))
	if '--file' in sys.argv:
		pos = sys.argv.index('--file')
		if len(sys.argv) > pos + 1:
			kwargs['file']=str(sys.argv[pos+1])
	if '--reps' in sys.argv:
		pos = sys.argv.index('--reps')
		if len(sys.argv) > pos + 1:
			reps=int(sys.argv[pos+1])

	for r in xrange(reps):
		main(*args,**kwargs)	
