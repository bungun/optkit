import sys
from optkit.tests import *

def main(*args, **kwargs):
	passing = True
	if '--linsys' in args: passing &= test_linsys(*args,**kwargs)
	if '--prox' in args: passing &= test_prox(*args, **kwargs)
	if '--proj' in args: passing &= test_projector(*args,**kwargs)
	if '--equil' in args: passing &= test_equil(*args,**kwargs)
	if '--norm' in args: passing &= test_normalizedprojector(*args,**kwargs)
	if '--block' in args: passing &= test_blocksplitting(*args,**kwargs)
	if '--pogs' in args: passing &= test_pogs(*args,**kwargs)
	print "all tests complete"
	if passing:
		print "all tests passed"

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
