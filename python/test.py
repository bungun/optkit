import sys
from optkit.tests import *

def main(m=30,n=20):
	# linsys_test()
	# prox_test()
	# projector_test()
	# blocksplitting_test(m,n)
	pogs_test(m,n)

if __name__== "__main__":
	if '--size' in sys.argv:
		pos = sys.argv.index('--size')
		if len(sys.argv) > pos + 2:
			m=int(sys.argv[pos+1])
			n=int(sys.argv[pos+2])
			main(m,n)
			exit()
	main()	
