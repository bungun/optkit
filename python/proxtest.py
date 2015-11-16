from optkit import *
from optkit import proxops
import numpy as np

f = FunctionVector(5)
proxops.print_function_vector(f)
print f.h_
proxops.push_function_vector(f)
proxops.print_function_vector(f)

f.b_ += 1.
f.c_ += 2.
f.d_ -= 0.45
proxops.push_function_vector(f)
proxops.print_function_vector(f)


rho = 1.
x = Vector(np.random.rand(5))
x_out = Vector(np.random.rand(5))

print x.py
print proxops.eval(f,x)
print x.py

print x_out.py
proxops.prox(f,rho,x,x_out)
print x_out.py


