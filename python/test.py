import sys
from optkit.tests import *

def main(args,m=30,n=20):
	print args
	if 'linsys' in args: linsys_test()
	if 'prox' in args: prox_test()
	if 'proj' in args: projector_test()
	if 'block' in args: blocksplitting_test(m,n)
	if 'pogs' in args: pogs_test(m,n)

if __name__== "__main__":
	args=[]
	if '--all' in sys.argv: args+=['linsys','prox','proj','block','pogs']
	else:
		if '--linsys' in sys.argv: args.append('linsys') 
		if '--prox' in sys.argv: args.append('prox') 
		if '--proj' in sys.argv: args.append('proj')
		if '--block' in sys.argv: args.append('block') 
		if '--pogs' in sys.argv: args.append('pogs') 
	if '--size' in sys.argv:
		pos = sys.argv.index('--size')
		if len(sys.argv) > pos + 2:
			m=int(sys.argv[pos+1])
			n=int(sys.argv[pos+2])
			main(args,m,n)
			exit()
	main(args)	
