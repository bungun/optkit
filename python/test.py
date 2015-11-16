from optkit import *
from optkit import ops
import numpy as np

print "TESTING:"
print "=======\n\n"


print "VECTOR METHODS"
print "--------------"


print "\nAllocation (int size=3) & printing"
print "---variable `a`---"
a=Vector(3)
print "numpy value:"
print a.py
print "c value:"
ops.print_var(a)


print "\nConstant addition: a += 1"
ops.add(1,a)
ops.print_var(a)

print "\nScalar multiplication: a *= 5"
ops.mul(5,a)
ops.print_var(a)

print "\nScalar division: a /= 2.3"
ops.div(2.3,a)
ops.print_var(a)

print "\nAllocation (from numpy array [1,2,3])"
print "---variable `b`---"
b=Vector(np.array([1,2,3]))
ops.print_var(b)

print "\nElementwise vector addition: a += b"
ops.add(b,a)
print "a:"
ops.print_var(a)
print "b:"
ops.print_var(b)

print "\nElementwise vector multiplication: a *= b"
ops.mul(b,a)
ops.print_var(a)

print "\nElementwise vector division: a /= b"
ops.div(b,a)
ops.print_var(a)

print "\nElementwise vector subtraction: a-= b"
ops.sub(b,a)
ops.print_var(a)

print "\nSync"
print "c to python"
ops.sync(a)
print a.py
ops.print_var(a)
print "python to c"
ops.sync(a,python_to_C=1)
print a.py
ops.print_var(a)

print "\nView"
print "---variable `a_view`---"
print "a_view = view(a, 0:2)"
c=ops.view(a,(0,2))
ops.print_var(c)

print "a_view *=2"
ops.mul(2,c)
print "a_view:"
ops.print_var(c)
print "a:"
ops.print_var(a)



print "\n\n"
print "MATRIX METHODS"
print "--------------"


print "\nAllocation (int size1=4,size2=3) & printing"
print "---variable `A`---"
A=Matrix(4,3)
print "numpy value:"
print A.py
print "c value:"
ops.print_var(A)


print "\nMatrix views:"

print "\n---variable `a_col`---"
print "column view: A[1]:"
a_col = ops.view(A,1,col=1)

print "a_col +=3.5"
ops.add(3.5,a_col)

print "a_col:"
ops.print_var(a_col)
print a_col.py
print "A:"
ops.print_var(A)
print A.py

print "\n---variable `a_row`---"
print "row view: A[1]:"
a_row = ops.view(A,1,row=1)

print "a_row -=5.34"
ops.sub(5.34,a_row)

print "a_row:"
ops.print_var(a_row)
print a_row.py
print "A:"
ops.print_var(A)
print A.py

print "\n---variable `a_diag'---"
print "diag view: A:"
a_diag = ops.view(A,diag=1)

print "a_diag /=2.1"
ops.div(2.1,a_diag)

print "a_diag:"
ops.print_var(a_diag)
print a_diag.py
print "A:"
ops.print_var(A)
print A.py

print "\nsync diagonal"
ops.sync(a_diag)
print "a_diag:"
ops.print_var(a_diag)
print a_diag.py

print "\nsubmatrix"
print "\n---variable `a_sub`---"
print "column view: A[0:1,1:2]:"
a_sub = ops.view(A,(0,1),(1,2),col=1)

print "a_sub /=3.5"
ops.div(3.5,a_sub)

print "a_col:"
ops.print_var(a_sub)
print a_sub.py
print "A:"
ops.print_var(A)
print A.py


print "\nAllocation (from 3x3 np.random.rand array)"
print "---variable `B`---"
B=Matrix(np.random.rand(3,3))
print "numpy value:"
print B.py
print "c value:"
ops.print_var(B)

print "\nMatrix scaling"
print "B *= 2"
ops.mul(2.,B)
print B.py
ops.print_var(B)


print "\nSync"
print "A += 1 (Python)"
A.py+=1

print "c to python"
ops.sync(A)
print A.py
ops.print_var(A)

print "A += 1 (Python)"
A.py+=1

print "python to c"
ops.sync(A,python_to_C=1)
print A.py
ops.print_var(A)


print "\nAllocation (from 4x3 np.random.rand array)"
print "---variable `C`---"
C=Matrix(np.random.rand(4,3))
print C.py
ops.print_var(C)


print "\nCopy"
print "---variable `D` = 0^{3x3}---"
D=Matrix(np.zeros((3,3)))
print D.py
ops.print_var(D)

print "D := memcopy(B)"
ops.copy(B,D)
print D.py
ops.print_var(D)


print "\n\n"
print "BLAS METHODS"
print "------------"

print "\nLEVEL 1"

print "\nVector-vector dot products"
print "(a,b)"
print ops.dot(a,b)


print "\n(a,a)"
print ops.dot(a,a)

print "\n2-norm: ||a||_2"
print ops.nrm2(a)

print "\n1-norm: ||a||_1"
print ops.asum(a)

print "\nBLAS axpy:"
print "a:"
ops.print_var(a)
print "b:"
ops.print_var(b)
print "b += 3a"
ops.axpy(3,a,b)
ops.print_var(b)

print "\nLEVEL 2"

print "\nBLAS gemv:"
print "---variable `d`---"
print "\nAllocation (from 4x1 np.random.rand array)"
d = Vector(np.random.rand(4))

print "d:"
ops.print_var(d)
print "a:"
ops.print_var(a)

print "d := 3Aa + 2d"
ops.gemv('N',3,A,a,2,d)
ops.print_var(d)
print d.py

print "d := 2.5Aa"
ops.gemv('N',2.5,A,a,0,d)
ops.print_var(d)
print d.py

print "\nLEVEL 3"

print "\nBLAS gemm:"
print "A:"
ops.print_var(A)
print "B:"
ops.print_var(B)


print "B := 2.13A^TA + 1.05B"
ops.gemm('T','N',2.13,A,A,1.05,B)
ops.print_var(B)
print B.py

print "B := A^TA"
ops.gemm('T','N',1,A,A,0,B)
ops.print_var(B)
print B.py


print "\n\n"
print "LINALG METHODS"
print "--------------"

print "---variable `E`---"
print "\nAllocation (from 10x10 np.random.rand array)"
F = np.random.rand(10,10)
E = Matrix(np.dot(F,F.T))


print E.py
Ecopy = Matrix(np.zeros((10,10)))
ops.copy(E,Ecopy)


print "---variable `x`---"
print "\nAllocation (from 10x1 np.random.rand array)"
x = Vector(np.random.rand(10))
print x.py


print "\nPython solve: E^-1 x"
pysol = np.linalg.solve(Ecopy.py,x.py)


print "\n Python cholesky"
LLT= np.linalg.cholesky(Ecopy.py)
print LLT


print "\nCholesky factorization"
ops.print_var(E)
	
ops.cholesky_factor(E)
print "E_LLT := chol(E) (lower triangular)"
ops.print_var(E)

print "L (py) - L (C):"
Ecopy.py[:]=(LLT - E.py)[:]
ops.print_var(Ecopy)

print "\nCholesky solve"

print "before"
print x.py

print "x := chol_solve(E_LLT,x)"
ops.cholesky_solve(E, x)

print "after"
print x.py

print "solve diff (C - py)"
print x.py - pysol

# # '''
# v = make_cvector()
# w = make_cvector()

# oklib.__vector_calloc(v, 10)
# oklib.__vector_calloc(w, 10)
# oklib.__vector_add_constant(v, 12.)
# oklib.__vector_add_constant(w, 5.)
# oklib.__blas_axpy(1., v , w)
# oklib.__vector_print(v)
# oklib.__vector_print(w)
# oklib.__vector_free(v)
# oklib.__vector_free(w)
# # '''


