from optkit.types import Vector
from optkit.kernels import *
from optkit.projector.direct import *
from numpy import nan, inf, ndarray, zeros, sqrt
from operator import add 

# TODO: lazy evaluation with Toolz?
class BlockVector(object):
	def __init__(self,m,n):
		# let m,n be ints or int tuples
		# then x and y would be Vectors or Vector tuples?
		self.size = m+n
		self.blocksizes = (m,n)
		self.vec = Vector(m+n)
		(self.x,self.y) = splitview(self.vec, m)
#
	def copy(self, other):
		if isinstance(other, Vector):
			copy(other, self.vec)
		elif isinstance(other, BlockVector):
			copy(other.vec, self.vec)
		else: raise TypeError("copy(BlockVector, x) only defined "
							  "for x of types optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))
#
	def add(self, other):
		if other==0: return
		if isinstance(other, (int,float)):
			add(other, self.vec)
		elif isinstance(other, Vector):
			add(other, self.vec)
		elif isinstance(other, BlockVector):
			add(other.vec, self.vec)
		else: raise TypeError("__add__(BlockVector, x) only defined "
							  "for x of types int,float,optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))
#
	def sub(self, other):
		if other==0: return
		if isinstance(other, (int,float)):
			sub(other, self.vec)
		elif isinstance(other, Vector):
			sub(other, self.vec)
		elif isinstance(other, BlockVector):
			sub(other.vec, self.vec)
		else: raise TypeError("__sub__(BlockVector, x) only defined "
							  "for x of types int,float,optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))
#
	def mul(self, other):
		if other == 1: return
		if isinstance(other, (int,float)):
			mul(other, self.vec)
		elif isinstance(other, Vector):
			mul(other, self.vec)
		elif isinstance(other, BlockVector):
			mul(other.vec, self.vec)
		else: raise TypeError("__mul__(BlockVector, x) only defined "
							  "for x of types int,float,optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))
#
	def div(self, other):
		if other == 1: return
		if isinstance(other, (int,float)):
			div(other, self.vec)
		elif isinstance(other, Vector):
			div(other, self.vec)
		elif isinstance(other, BlockVector):
			div(other.vec, self.vec)
		else: raise TypeError("__div__(BlockVector, x) only defined "
							  "for x of types int,float,optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))
	def __str__(self):
		return reduce(add,["\n\t\t{}: {}".format(k,self.__dict__[k]) for k in self.__dict__])

class BlockMatrix(object):
	pass


class SolverMatrix(object):
	def __init__(self, A):
		if not isinstance(A, (ndarray, Matrix)):
			raise TypeError("SolverMatrix initialization requires "
							"a numpy.ndarray or optkit.Matrix\n."
							"Provided: {}".format(type(A)))
		if isinstance(A, ndarray): 
			self.mat = Matrix(A)
		elif isinstance(A, Matrix): 
			self.mat = A

		self.shape = A.shape
		self.equilibrated = False
		self.normalized = False
		self.norm = nan
	def __str__(self):
		return str(self.__dict__)

class SolverState(object):
	def __init__(self, *data):
		for item in data:
			if isinstance(item,SolverMatrix):
				self.A=item
			if isinstance(item,Projector):
				self.Proj=item
			if isinstance(item,ProblemVariables):
				self.z=item
		if not 'A' in self.__dict__: self.A=None
		if not 'Proj' in self.__dict__: self.Proj=None
		if not 'z' in self.__dict__: self.z=None


	def __str__(self):
		return str(self.__dict__)




class ProblemVariables(object):
	def __init__(self,m,n):
		self.primal = BlockVector(m,n)
		self.primal12 = BlockVector(m,n)
		self.dual = BlockVector(m,n)
		self.dual12 = BlockVector(m,n)
		self.prev = BlockVector(m,n)
		self.temp = BlockVector(m,n)
		self.de = BlockVector(m,n)
		self.size=m+n
		self.blocksizes=(m,n)
	def __str__(self):
		return reduce(add,["\n\t{}: {}".format(k,self.__dict__[k]) for k in self.__dict__])

class OutputVariables(object):
	def __init__(self,m,n):
		self.x = np.zeros(n)
		self.y = np.zeros(m)
		self.mu = np.zeros(n)
		self.nu = np.zeros(m)
	def __str__(self):
		return reduce(add,["{}: {}\n".format(k,self.__dict__[k]) for k in self.__dict__])

class Objectives(object):
	def __init__(self):
		self.p = inf
		self.d = -inf
		self.gap = inf
	def __str__(self):
		return str(self.__dict__)

class Residuals(object):
	def __init__(self):
		self.p = inf
		self.d = inf
		self.gap = inf
	def __str__(self):
		return str(self.__dict__)

class Tolerances(object):
	def __init__(self, m, n, **options):
		self.p = 0
		self.d = 0
		self.gap = 0
		self.reltol = options['rtol'] if 'rtol' in options else 1e-3
		self.abstol = options['atol'] if 'atol' in options else 1e-4
		self.atolm = sqrt(m) * self.abstol
		self.atoln = sqrt(n) * self.abstol
		self.atolmn = sqrt(m*n) * self.abstol
	def __str__(self):
		return str(self.__dict__)


class AdaptiveRhoParameters(object):
	def __init__(self, **kwargs):
		self.RHOMAX = kwargs['rhomax'] if 'rhomax' in kwargs else 1e4
		self.RHOMIN = kwargs['rhomin'] if 'rhomin' in kwargs else 1e-4
		self.DELTAMAX = kwargs['deltamax'] if 'deltamax' in kwargs else 1.05
		self.DELTAMIN = kwargs['deltamin'] if 'deltamin' in kwargs else 2.
		self.GAMMA = kwargs['gamma'] if 'gamma' in kwargs else 1.01
		self.KAPPA = kwargs['kappa'] if 'kappa' in kwargs else 0.9
		self.TAU = kwargs['tau'] if 'tau' in kwargs else 0.8
		self.delta = self.DELTAMIN
		self.l = 0.
		self.u = 0.
		self.xi = 1.
	def __str__(self):
		return str(self.__dict__)



class SolverInfo(object):
	def __init__(self):
		self.err = 0
		self.converged = False
		self.k = 0
		self.obj = inf
		self.rho = None

	def update(self, **info):
		print info
		if 'err' in info: self.err = info['err']
		if 'converged' in info: self.converged = info['converged']
		if 'k' in info: self.k = info['k']
		if 'obj' in info: self.obj = info['obj']
		if 'rho' in info: self.rho = info['rho']

	def __str__(self):
		return str(self.__dict__)

class SolverSettings(object):
	ALPHA = 1.7
	MAXITER = 2000
	RHO = 1
	ATOL = 1e-4
	RTOL = 1e-3
	ADAPT = True
	GAPSTOP = False
	WARM = False
	VERBOSE = 2
	def __init__(self, **options):
		self.alpha = self.ALPHA
		self.maxiter = self.MAXITER
		self.rho = self.RHO
		self.abstol = self.ATOL
		self.reltol = self.RTOL
		self.adaptive = self.ADAPT
		self.gapstop = self.GAPSTOP
		self.warmstart = self.WARM
		self.verbose = self.VERBOSE
		self.update(**options)
	def update(self, **options):
		if 'alpha' in options: self.alpha = options['alpha'] 
		if 'maxiter' in options: self.maxiter = options['maxiter']  
		if 'rho' in options: self.rho = options['rho'] 
		if 'abstol' in options: self.abstol = options['abstol']  
		if 'reltol' in options: self.reltol = options['reltol'] 
		if 'adaptive' in options:self.adaptive = options['adaptive'] 
		if 'gapstop' in options: self.gapstop = options['gapstop']  
		if 'warmstart' in options: self.warmstart = options['warmstart'] 
		if 'verbose' in options: self.verbose = options['verbose'] 
		self.validate()
	def validate(self):
		pass
	def __str__(self):
		return str(self.__dict__)





