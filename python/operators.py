	class AbstractMatrix(object):
		pass		

class DenseMatrix(AbstractMatrix):
	def __init__(self, A):
		(self.m, self.n) = self.shape = A.shape
		self.mat = A
		self.equilibrated = False
		self.normalized = False	

class SparseMatrix(AbstractMatrix):
	def __init__(self, A):
		(self.m, self.n) = self.shape = A.shape
		self.mat = A
		self.tranpose = A.T
		self.equilibrated = False
		self.normalized = False	

class IdentityMatrix(AbstractMatrix):
	def __init__(self, n):
		(self.m, self.n) = self.shape = (n,n)

class DiagonalMatrix(AbstractMatrix):
	def __init__(self, x):
		(self.m, self.n) = self.shape = (len(x),len(x))
		self.diag = x








# class BlockIdentityMatrix(AbstractMatrix):
# 	def __init__(self, m , n, **lengths_or_pointers)
# 		self.m = m
# 		self.n = n
# 		self.shape = (m,n)
# 		if 'pointers' in lengths_or_pointers:
# 			pass
# 		elif 'lengths' in lengths_or_pointers:
# 			pass

class UpsamplingMatrix(AbstractMatrix):
	def __init__(self, indexmap):
		(self.m, self.n) = self.shape = (len(indexmap),int(max(indexmap)))
		self.indexmap = u

class DownsamplingMatrix(AbstractMatrix):
	def __init__(self, indexmap):
		(self.m, self.n) = self.shape = (int(max(indexmap)), len(indexmap))
		self.indexmap = indexmap
	def __mul__(self, other):
		pass

class ZeroMatrix(AbstractMatrix):
	def __init__(self, m, n):
		(self.m, self.n) = self.shape = (m,n)


# class BlockMatrix(object):
# 	def __init__(self, m, n):
# 		if isinstance(m,int) and isinstance(n,int): 
# 			raise TypeError("At least one argument must be a list")
# 		if not isinstance(m, (int,list)): 
# 			raise TypeError("Input arguments must be int or list"):
# 		if not isinstance(m, (int,list)): 	
# 			raise TypeError("Input arguments must be int or list"):

# 		self.m = m if isinstance(m,list) else [m]
# 		self.n = n if isinstance(m,list) else [n]
# 		self.M = sum(m)
# 		self.N = sum(n)
# 		self.skinny = self.M >= self.N
# 		self.square = self.M == self.N

# 		self.blocks = {}
# 		self.block_shapes = {}
# 		for i in xrange(self.m_blocks):
# 			for j in xrange(self.n_blocks):
# 				self.blocks["{}{}".format(i,j)]=ZeroMatrix(m[i],n[j])
# 				self.block_shapes["{}{}".format(i,j)]=(m[i],n[j])


# 	def set_block(self, address, A)
# 		if not A.shape == self.block_shapes[address]:
# 			raise ValueError("cannot assign input to block: "
# 							  "incompatible size")


# B = 
# [ A 	0
#   -I    U]

# lets say A 10k x 50k
# 		 0 10k x 250
# 		 I 50k x 50k
# 		 U 50k x 250

# B'B = 

# [A' -I     [ A   0
#  0   U']    -I   U ]

# [A'A + I    -U
#    -U'	    U'U	 ]

# B'B+I = 

# [A'A + 2I    -U
#    -U'	    U'U	+ I ]


# BB' =

# [ A   0     [ A'   -I
#  -I   U]      0     U' ]

# [ AA'  -A
#   -A'  I + UU']

# BB' + I =

class MatrixOperator(object):
	pass


# mul(a, input, b, output)
class DenseOperator(MatrixOperator):
	def __init__(self, A):
		if A is not None:
			self.data = A
			self.shape = A.shape
			self.mul = gemv_reordered('N',A)
			self.mul_t = gemv_reordered('N',A)
		else:
			self.data=None
			self.shape=None
			self.mul=None
			self.mult_t=None
	def get_adjoint(self):
		return ArbitraryOperator((self.shape[1],self.shape[0]), self.mul_t, self.mul)	
		# adj = DenseOperator(None)
		# adj.data = None
		# adj.shape = (self.shape[1], self.shape[0])
		# adj.mul = self.mul_t
		# adj.mul_t = self.mul
	def get_inverse(self):
		pass

class GramianOperator(DenseOperator):
	def __init__(self, A, AA, transpose=True):
		self.A = A
		self.data = AA
		self.transpose = transpose
		self.shape = (A.shape[0],A.shape[0]) if transpose else (A.shape[1],A.shape[1])
			self.mul = gemv_reordered('N',AA)
			self.mul_t = gemv_reordered('N',AA)

	# def as_dense(self):
	# 	if self.transpose:
	# 		return DenseOperator(self.data.T*self.data)
	# 	else
	# 		return DenseOperator(self.data*self.data.T)





# DenseOperator(A):
# 	self.A = A
# 	self.shape = A.shape
# 	self.invertible = A.shape[0]==A.shape[1]
# 	self.mul(a, input, b, output) = gemv_reorder('N', A)
# 	self.mul_t(a, input, b, output) = gemv_reorder('T', A)
# 	self.add(DenseOperator)=DenseOperator
# 	self.add(SparseOperator)=DenseOperator
# 	self.add(IdentityOperator)=DenseOperator
# 	self.add(DiagonalOperator)=DenseOperator
# 	self.add(ZeroOperator)=self
# 	self.AAt = DenseOperator(A*A')
# 	self.AtA = DenseOperator(A'*A)


class SparseOperator(MatrixOperator):
	def __init__(self, A):
		if A is not None:
			self.data = A
			self.data_adj = A.T
			self.shape = A.shape
			# self.mul = gemv_reordered('N',A)
			# self.mul_t = gemv_reordered('N',A)
		else:
			self.data=None
			self.data_adj=None
			self.shape=None
			self.mul=None
			self.mult_t=None
	def get_adjoint(self):
		return ArbitraryOperator((self.shape[1],self.shape[0]), self.mul_t, self.mul)	


class ScaledIdentityOperator(MatrixOperator):
	def __init__(self, n, alpha=1):
		self.shape = (n,n)

		pass

	def get_adjoint(self):
		return self

	def get_inverse(self):
		pass

class DiagonalOperator(MatrixOperator):
	pass

class ZeroOperator(MatrixOperator):
	pass

class DyadicOperator(MatrixOperator):
	pass

class UpsamplingOperator(MatrixOperator):
	pass


class DyadicOperator(MatrixOperator):
	pass

class ChainedOperator(MatrixOperator):
	pass

class CompositeOperator(MatrixOperator):
	pass


class ArbitraryOperator(MatrixOperator):
	def __init__(self, shape, mul, mul_t):
		self.shape = shape
		self.mul = mul
		self.mul_t = mul_t

	def get_adjoint(self):
		return ArbitraryOperator((self.shape[1],self.shape[0]), self.mul_t, self.mul)










IdentityOperator(n):
	self.shape = (n,n)
	self.mul(a, input, b, output) = aipx
	self.mul_t = self.mul
	self.add(IdentityOperator)=ScaledIdentityOperator()
	self.add(DiagonalOperator)=DiagonalOperator()
	self.add(ZeroOperator)=self

ScaledIdentityOperator()

DiagonalOperator(x):
	self.diag = x
	self.shape = (x.size, x.size)
	self.mul(a, input, b, output) = sbmv_reorder(x)
	self.mul_t = self.mul
	self.mul_inplace(a, input) = mul(self,x)
	self.add(DiagonalOperator)=DiagonalOperator


BlockDiagonalOperator(x):
	self.shape
	self.block_sizes
	self.add(DiagonalOperator)=BlockDiagonalOperator
	self.add(IdentityOperator)=BlockDiagonalOperator



SquareBlockDiagonal

ZeroOperator(m):
	self.shape = (n,n)
	self.mul(a, input, b, output) = axpby(0)
	self.gramian = self


ChainedOperator(*Ops):
	self.shape = (Ops[0].shape[0], Ops[-1].shape[1])
	self.ops = Ops
	self.intermediates = []
	for op in xrange(len(Ops)-1):
		self.intermediates.append(Vector(op.shape[1]))
	self.mul(a, input, b, output):
		ops[-1].mul(a, input, 0, self.intermediates[0])
		for i,op in enumerate(self.ops.__reversed__()):
			if i==0:continue
			op.mul(self.intermediates[-(i+1)])

ChainedSquareOps(*Ops)
	self.shape = Ops[0].shape
	self.ops = Ops
	self.mul(a, input, b, output):
		ops[-1].mul(a, input, b, output)
		for op in self.ops.__reversed__():
			op.mul(1, output, 1, output)



CompositeOperator(*Ops):
	self.shape = Ops[0].shape
	self.ops = Ops
	self.mul(a, input, b, output):
		for i,op in enumerate(ops):
			if i==0:
				op.mul(a, input, b, output)
			else:
				op.mul(a, input, 1, output)


DyadicOperator(Op1, Op2):
	self.shape = (Op1.shape[0],Op2.shape[1])
	self.intermediate = Vector(Op1.shape[0])
	self.mul(a, input, b, output) = 
		Op.1.mul(a, input, 0, self.intermediate)
		Op.2.mul(1, self.intermediate, 1, output)
	self.mul_t = self.mul

UpsamplingOperator(lengths):
	self.lengths = lengths
	self.shape = (sum(lengths),len(lengths))
	self.mul(a, input, b, output) = clone
	self.mul_t(a, input, b, output) =  sum
	self.mul_t_inplace
	self.AAt = ChainedOperator()
	self.AtA = DiagonalOperator(self.lengths)
	self.transpose:
		OpT=GenericOperator():
		OpT.shape=(self.shape[1],self.shape[0])
		OpT.mul=self.mul_t
		OpT.mul_t = self.mul
		return OpT


DownsamplingOperator(lengths):
	

GenericOperator():
	self.shape
	self.mul
	self.mul_t
	self.transpose:
		.shape = (self.shape[1], self.shape[0])
		.mul = self.mul_t
		.mul_t = self.mul



# B:=
# B['11']=DenseOperator(A)
# B['21']=ScaledIdentityOperator(-1,nx)
# B['12']=ZeroOperator(nx,nz)
# B['22']=UpsamplingOperator(u)



# [ AA'		-A     ]
# [ -A'		UU' + I]

# BB' :=
# 	BB['11']=DenseOperator(AA')
# 	BB['21']=GenericOperator()
# 			.shape = B['11'].shape.reverse()
# 			.mul = compose(signflip, B['11'].mul_t)
# 			.mul_t = compose(signflip, B['11'].mul)
# 	BB['12']=BB['21'].transpose
# 	BB['22']=CompositeOperator(B['22'].AAt + ScaledIdentityOperator(1,nx))

# [ I + AA'  		-A
#    -A' 		 2I + UU']

# BB' + I  :=
# 	BB['11']=DenseOperator(AA')+ ScaledIdentityOperator(1,ny)
# 	BB['21']=GenericOperator()
# 			.shape = B['11'].shape.reverse()
# 			.mul = compose(signflip, B['11'].mul_t)
# 			.mul_t = compose(signflip, B['11'].mul)
# 	BB['12']=BB['21'].transpose
# 	BB['22']=CompositeOperator(B['22'].AAt + ScaledIdentityOperator(2,nx))


# B'B = 

# [A'A + I    -U
#    -U'	    U'U	 ]

# B'B :=
# 	B'B['11']=DenseOperator(A'A)+IdentityOperator(ny)
# 	B'B['21']=GenericOperator():
# 			.shape = B['21'].shape.reverse()
# 			.mul = compose(signflip, B['21'].mul_t)
# 			.mul_t = compose(signflip, B['21'].mul)
# 	B'B['12']=BB['21'].transpose
# 	B'B['22']=B['22'].AtA

# B'B + I =

# [A'A + 2I    -U
#    -U'	    U'U	+ I ]

# B'B + I:=
# 	B'B['11']=DenseOperator(A'A)+ScaledIdentityOperator(2,ny)
# 	B'B['21']=GenericOperator():
# 			.shape = B['21'].shape.reverse()
# 			.mul = compose(signflip, B['21'].mul_t)
# 			.mul_t = compose(signflip, B['21'].mul)
# 	B'B['12']=BB['21'].transpose
# 	B'B['22']=B['22'].AtA + IdentityOperator(nz)



CholeskyInverseOperator(A::DenseOperator) 
	self.shape = A.shape
	self.L = cholesky_factor(A)
	self.forwardsolve = cholesky_solve(self.L)
	self.backsolve = cholesky_solve(self.L)
	self.mul(a, input, b, output):
		if b != 0 warn("using b=0")
		self.forwardsolve(output)
		self.backsolve(output)		
	self.mul_t = self.mul

# inv(A).solve(scratch, vector):
#	inv(A).mul(vector)

BlockOperator()
	self.m_list
	self.n_list
	self.shape
	self.m_blocks
	self.n_blocks
	self.blocks={}
	self.get_block() #return compatible sized Zero if not in dictionary
	self.get_adjoint
	self.mul
	self.mul_adj
	self.cost
	self.cost_adj


SquareBlockOperator()
	self.shape
	self.A =
	self.B =
	self.C =
	self.D =
	# this should be symbolic until/unless forced
	self.SchurA = self.D - self.C * inv(self.A) * self.B

	# // Add(D, Chain(-I, C, inv(A), B))

	# OR

	self.SchurD = self.A - self.C * inv(self.A) * self.B
 
	# // Add(A, Chain(-I, C, inv(D), B))




# MATRIX INVERSION LEMMA

# inv [A U   =   [I 		0  [  inv(A-Uinv(C)V)      0
#  	   V C]		 -inv(C)V   I]         0             inv(C)

# In general:

# BlockInverse(M):
# 	M.A
# 	M.U
# 	M.V
# 	M.C


# 	D = BlockDiagonalOperator(2, n1, n2) 
# 		D.11 = inv(M.A - M.U*inv(M.C)*M.V)
# 		D.22 = inv(M.C)

# 	# Assuming M.A is dense, 
# 	# D.22 =  inv(M.C)
# 	# D.11 =  inv(DenseOperator - Op * Op * Op) => inv(DenseOperator)
# 	# 	   =  CholeskyInverseOperator(M.A + M.U*D.22.M.V)

# 	L = BlockLowerDiagonalOperator(2, m, n)
# 		L.11 = IdentityOperator
# 		L.21 = compose(sign_flip, D.22.mul, M.V.mul)
# 		L.12 = ZeroOperator
# 		L.22 = IdentityOperator


# 	U = BlockUpperDiagonalOperator(2, m, n)
# 		U.11 = IdentityOperator
# 		U.21 = ZeroOperator
# 		U.12 = compose(sign_flip, M.U.mul, D.22.mul)
# 		U.22 = IdentityOperator



BlockInverseOperator(M::BlockOperator)
	self.D = DiagonalBlockOperator()
		.D.22 = inv(M.D)
		.D.11 = inv(M.SchurD)
		OR
		.D11 = inv(M.D)
		.D22 = inv(M.SchurD)

	self.L = LowerBlockOperator()
		.L.11=IdentityOperator()
		.L.21=Chain(-I,.D.22,M.V)
		.L.22=IdentityOperator()
	self.U = UpperBlockOperator()
		.U.11=IdentityOperator()
		.U.12=Chain(-I,M.U,.D.22)
		.U.22=IdentityOperator()

# In particular:

# BlockInverse(M'M + I ):
# Start with:
# 	M.A = A
#	M.B = Zero
#	M.C = -I
#   M.D = U


# MtMi --- allocate memory for diagonal blocks

#	MtMi.A = A'A + 2I == _A_
#	MtMi.B = Chain(-I,U')
#	MtMi.C = Chain(-I,U)
#	MtMi.D = U'U + I
#	MtMi.SchurD = (_A_ - U' inv(U'U+I) U)

# MtMi is a BlockOperator with
# .A := DenseOperator. NEEDS ALLOCATION
# .B := UpsamplingOperator. ALREADY ALLOCATED
# .C := UpsamplingOperator.Transpose (or DownsamplingOperator). ALREADY ALLOCATED
# .D := DiagonalOperator. NEEDS ALLOCATION
# .SchurD := DenseOperator - HOLD UNTIL NEEDED
# .SchurA := DenseOperator - HOLD UNITIL NEEDED

# inv(MtMi) is a BlockInverseOperator with
# .L := BlockLowerTriangular
#	.L11 = I
#	.L21 = Chain(-I, Chain(-I, U'), D22) -> NON-GENERIC CHAIN
#	.L22 = I
# .D := BlockDiagonal
#	.D22 = inv(MtMi.D) := DiagonalInverseOperator
#	.D11 = inv(MtMi.SchurD) := CholeskyInverseOperator
# .U := BlockUpperTriangular
#	.U11 = Identity
#	.U12 = Chain(-I, .D22, Chain(-I,U)) = Chain(.D22, U') -> GENERIC CHAIN
#	.U22 = Identity


# inv(MtMi).solve(scratch, vector):
#	inv(MtMi).U.mul(1, scratch, 1, vector):
#		.U12.mul(1, vector.2, 1, vector.1), i.e., sum then scale
#		
#	inv(MtMi).D.mul(1, scratch, 1, vector):
#		.D22.solve(vector.2)
#		.D11.solve(vector.1)
#
#	inv(MtMi).L.mul(1, scratch, 1, vector):
#		scratch.2 = copy(vector.2)
#		.L21.mul(1, vector.1, 0, vector.2), i.e. scale then copy -> WHEN THIS IS ZERO, SHOULD USE MEMSET IN BLOCKS
#		vector.2 += scratch.2
#		



# RECURSIVE:
#
# M0 = [A0, B0; C0, D0]
# M1 = [M0, B1; C1, D1]
#
#
# MtMi.A -> BlockOperator (instead of DenseOperator)
# inv(MtMi).D11 -> BlockInverseOperator (instead of CholeskyInverseOperator)
#
#
#
#


# I GUESS WE'LL ALSO NEED HSTACK AND VSTACK FOR NON-INVERTED BLOCKS?

#  DEFINE BLOCK VECTORS WITH VIEWS 
#  GO THROUGH PARSE TREE TO:
#	1. see which memory needs to be allocated
#	2. which memory views need to be allocated /
#	3. which operations are better resolved (eg, Dense + Sparse -> Dense; or, SchurA-> unresolved if SchurD better)
#
#	Maybe need flags for operations: resolved, usable, allocated, etc.
#
#
#
#

#
# So now, consider matrix G
#
#
#	G = [ A_targ  0     0
#         A_ntc   0     0
#		  A_ntf   -U_y  0
#         I       0     -U_x ]
#
#	
#  G := Stack{
#		Dense(A_targ)
#		Dense(A_ntc)
#		Dense(A_ntf) + chain(-I,Upsampling(U_y))
#		I + chain(-I,Upsampling(U_x))}
#
#
#  [Stack-Add vs. Add-Stack]
#
#	equivalently,
#
#	:= Add{
#		Stack{Dense, Dense, Dense, I}
#		Stack{Zero, Zero, Chain(-I,Uy), Zero}
#		Stack{Zero, Zero, Zero, Chain(-I, Ux)}
#
#	G'G + I =
#
#	[  A_t'A_t + A_ntc'A_ntc + A_ntf'A_ntf + 2I  	-A_ntf'U_y		  	 -U_x
#   					-U_y'A_ntf					  U_y'U_y  + I          0
#							-U_x'						0  		       U_x'U_x +I ]
#
#
# 	Dense + Dense/Outer + Dense/Outer + 2Id      -Id * Dense * Upsample     -Id*Upsample 				
#   -Id * Downsample * Dense                          Diag + Id   				  Zero
#        -Id * Downsample  								Zero  					Diag + Id
#
#
#
#   Reducing:
#
#  	Dense (but don't form)	  	-Id * Dense * Upsample  	-Id*Upsample
# 	-Id * Downsample * Dense  	Diag 						Zero
#	-Id * Downsample 			Zero 						Diag
# 	
#
#   Parse 1:
#   	M.A=Dense, inversion cost = nx^3; inverse apply cost: nx^2
#		M.B=-Id*Stack(Downsample*Dense, Downsample)
#       M.C=M.B'
#       M.D=Block {Diag,Diag}  -> from this, infer SchurD > SchurA
#		M.SchurD = A -  B invD C ::= Dense - [Dense*Upsample  -Upsample]  [Diag * Downsample * Dense ; Diag * Downsample]
#       M.L21 = Stack(Downsample*Dense,Downsample)	
#       M.L12 = Add(Dense*Upsample,Upsample)
#
#  Parse 2:
#		M.A=Block, direct inversion cost = (nx+nS)^3; inverse apply cost:  nx^2
#				   SWMF inversion cost = nx^3 ; inverse apply cost: nx^2 + nS + 2*ny*nS
#				   => take direct
#		M.B=Upsample: apply cost: ny
#		M.C=Downsample = M.B': apply cost: ny
# 		M.D=Diagonal: apply cost
#		
#
#

class BlockProjector(object):
	def __init__(self):
		pass	