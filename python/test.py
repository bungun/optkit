import sys
from optkit.tests import *

def main(args,m=30,n=20):
	print args
	if '--linsys' in args: test_linsys()
	if '--prox' in args: test_prox()
	if '--proj' in args: test_projector()
	if '--equil' in args: test_equil()
	if '--norm' in args: test_normalizedprojector()
	if '--block' in args: test_blocksplitting(m,n)
	if '--pogs' in args: test_pogs(m,n)

if __name__== "__main__":
	args=[]
	if '--all' in sys.argv: 
		args+=['--linsys','--prox','--proj','--equil',
			'--norm','--block','--pogs']
	else:
		args+=sys.argv

	if '--size' in sys.argv:
		pos = sys.argv.index('--size')
		if len(sys.argv) > pos + 2:
			m=int(sys.argv[pos+1])
			n=int(sys.argv[pos+2])
			main(args,m,n)
			exit()
	main(args)	
