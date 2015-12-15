from optkit.defs import FLOAT_CAST
from optkit.types import Vector
from optkit.utils.pyutils import var_assert
from optkit.kernels import *
from optkit.projector.direct import *
from numpy import nan, inf, ndarray, zeros
from operator import add 

class BlockVector(object):
	def __init__(self,m,n):
		# let m,n be ints or int tuples
		# then x and y would be Vectors or Vector tuples?
		self.size = m+n
		self.blocksizes = (m,n)
		self.vec = Vector(m+n)
		(self.x,self.y) = splitview(self.vec, m)
#
	def isvalid(self):
		assert isinstance(self.size,int)
		for s in self.blocksizes:
			assert isinstance(s,int)
		assert sum(self.blocksizes)==self.size
		assert isinstance(self.vec,Vector)
		assert self.vec.isvalid()
		assert self.vec.size==self.size
		assert isinstance(self.x,Vector)
		assert isinstance(self.y,Vector)
		assert self.x.isvalid()
		assert self.y.isvalid()
		assert self.x.size==blocksizes[0]
		assert self.y.size==blocksizes[1]
		return True


	def dot(self,other):
		if isinstance(other, Vector):
			return dot(self.vec,other)
		elif isinstance(other, BlockVector):
			return dot(self.vec,other.vec)
		else: raise TypeError("copy_from(BlockVector, x) only defined "
							  "for x of types optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))

	def copy_to(self, other):
		if isinstance(other, Vector):
			copy(self.vec,other)
		elif isinstance(other, BlockVector):
			copy(self.vec,other.vec)
		else: raise TypeError("copy_from(BlockVector, x) only defined "
							  "for x of types optkit.Vector "
							  "or BlockVector.\n Provided: "
							  "{}".format(type(other)))

	def copy_from(self, other):
		if isinstance(other, Vector):
			copy(other, self.vec)
		elif isinstance(other, BlockVector):
			copy(other.vec, self.vec)
		else: raise TypeError("copy_from(BlockVector, x) only defined "
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
			self.orig = FLOAT_CAST(A)
		elif isinstance(A, Matrix): 
			self.orig = FLOAT_CAST(A.py)

		self.mat = Matrix(self.orig)
		self.shape = A.shape
		self.equilibrated = False
		self.normalized = False
		self.norm = nan
	def __str__(self):
		return str(self.__dict__)
	def isvalid(self):
		assert isinstance(self.orig,ndarray)
		assert isinstance(self.mat, optkit.Matrix)
		assert self.mat.isvalid()
		for item in ['shape','equilibrated','normalized','norm']:
			assert self.__dict__.has_key(item)
			assert self.__dict__[item] is not None
		assert self.shape == self.mat.shape
		assert self.shape == self.orig.shape
		return True

class SolverState(object):
	def __init__(self, *data):
		for item in data:
			if isinstance(item,SolverMatrix):
				self.A=item
			if isinstance(item,Projector):
				self.Proj=item
			if isinstance(item,ProblemVariables):
				self.z=item
			if isinstance(item,float):
				self.rho=item
		if not 'A' in self.__dict__: self.A=None
		if not 'Proj' in self.__dict__: self.Proj=None
		if not 'z' in self.__dict__: self.z=None
		if not 'rho' in self.__dict__: self.rho=None

	def __str__(self):
		return str(self.__dict__)

	def isvalid(self):
		for item in ['A','Proj','z','rho']:
			assert self.__dict__.has_key(item)
		assert self.A is None != var_assert(self.A,type=SolverMatrix)
		assert self.Proj is None != var_assert(self.Proj,type=Projector)
		assert self.z is None != var_assert(self.z,type=ProblemVariables)
		assert self.rho is None != var_assert(self.rho,type=float)
		assert self.A.shape == self.Proj.A.shape
		assert self.A.shape == self.z.blocksizes
		return True


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
	def isvalid(self):
		assert isinstance(self.size,int)
		assert isinstance(self.blocksizes,tuple)
		for s in self.blocksizes:
			assert isinstance(s,int)
		for item in ['primal','primal12','dual','dual12',
					'prev','temp','de']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=BlockVector)
			assert self.__dict__[item].size==self.size
			assert self.__dict__[item].blocksizes==self.blocksizes			
		return True
class OutputVariables(object):
	def __init__(self,m,n):
		self.x = np.zeros(n)
		self.y = np.zeros(m)
		self.mu = np.zeros(n)
		self.nu = np.zeros(m)
	def __str__(self):
		return reduce(add,["{}: {}\n".format(k,self.__dict__[k]) for k in self.__dict__])
	def isvalid(self):
		for item in ['x','y','mu','nu']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=ndarray)
		assert self.x.size==self.mu.size
		assert self.y.size==self.nu.size		
		return True


class Objectives(object):
	def __init__(self):
		self.p = inf
		self.d = -inf
		self.gap = inf
	def __str__(self):
		return str(self.__dict__)
	def isvalid(self):
		for item in ['p','d','gap']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=float)		
		return True

class Residuals(object):
	def __init__(self):
		self.p = inf
		self.d = inf
		self.gap = inf
	def __str__(self):
		return str(self.__dict__)
	def isvalid(self):
		for item in ['p','d','gap']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=float)		
		return True


class Tolerances(object):
	def __init__(self, m, n, **options):
		self.p = 0.
		self.d = 0.
		self.gap = 0.
		self.reltol = options['rtol'] if 'rtol' in options else 1e-3
		self.abstol = options['atol'] if 'atol' in options else 1e-4
		self.atolm = m**0.5 * self.abstol
		self.atoln = n**0.5 * self.abstol
		self.atolmn = (m*n)**0.5 * self.abstol
	def __str__(self):
		return str(self.__dict__)
	def isvalid(self):
		for item in ['p','d','gap','reltol','abstol',
					'atolm','atoln','atolmn']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=float)		
		return True


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
	def isvalid(self):
		for item in ['RHOMAX','RHOMIN','DELTAMAX','DELTAMIN',
					'GAMMA','KAPPA','TAU','delta','l','u','xi']:
			assert self.__dict__.has_key(item)
			assert var_assert(self.__dict__[item],type=float)
		return True


class SolverInfo(object):
	def __init__(self):
		self.err = 0
		self.converged = False
		self.k = 0
		self.obj = inf
		self.rho = None
		self.setup_time = inf
		self.solve_time = inf

	def update(self, **info):
		if 'err' in info: self.err = info['err']
		if 'converged' in info: self.converged = info['converged']
		if 'k' in info: self.k = info['k']
		if 'obj' in info: self.obj = info['obj']
		if 'rho' in info: self.rho = info['rho']

	def __str__(self):
		return str(self.__dict__)

	def print_status(self):
		if self.err > 0:
			status_string = "Error"
		elif self.converged:
			status_string = "Converged"
		else:
			status_string = "Inaccurate"
		print "***********************************************"
		print "\tSOLVER STATUS: ", status_string
		print "\tOBJECTIVE: ", self.obj
		print "\tITERATIONS: ", self.k
		print "\tSETUP TIME: ", self.setup_time
		print "\tSOLVE TIME: ", self.solve_time
		print "***********************************************\n"

	def isvalid(self):
		for item in ['err','converged','k','obj','rho']:
			assert self.__dict__.has_key(item)
		assert var_assert(self.obj,self.rho,type=(float,int,long))
		assert var_assert(self.converged,type=(bool,int,long))
		assert var_assert(self.err,self.k,type=(int,long))
		return True

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
	RESUME = False
	DEBUG = False
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
		self.resume = self.RESUME 
		self.debug = self.DEBUG
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
		if 'resume' in options: self.resume = options['resume']
		if 'debug' in options: self.debug = options['debug']
		self.validate()
	def validate(self):
		pass
	def __str__(self):
		return str(self.__dict__)
	def isvalid(self):
		for item in ['ALPHA,','MAXITER','RHO','ATOL','RTOL',
			'ADAPT','GAPSTOP','WARM','VERBOSE', 'RESUME', 'DEBUG',
			'alpha','maxiter', 'abstol','reltol','adaptive',
			'gapstop','warmstart', 'verbose', 'resume', 'debug']:
			assert self.__dict__.has_key(item)
		assert var_assert(self.ALPHA,self.alpha,self.RHO,self.rho,
						self.ATOL, self.abstol,self.RTOl,
						self.reltol,type=(int,long,float))
		assert var_assert(self.VERBOSE,self.verbose,type=(int,long))
		assert var_assert(self.ADAPT,self.adaptive,self.GAPSTOP,
						self.gapstop,self.WARM,self.warmstart,
						self.RESUME, self.resume, self.DEBUG,
						self.debug, type=(int,long,bool))
		return True




