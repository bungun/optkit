from os import getenv

if getenv('OPTKIT_IMPORT_BACKEND', 0) > 1:
	from optkit.api import backend

from optkit.api import set_backend

# Types 
from optkit.api import Vector
from optkit.api import Matrix
from optkit.api import FunctionVector

# Linsys calls 
from optkit.api import set_all
from optkit.api import copy
from optkit.api import view
from optkit.api import sync
from optkit.api import print_var
from optkit.api import add
from optkit.api import sub
from optkit.api import mul
from optkit.api import div
from optkit.api import elemwise_inverse
from optkit.api import elemwise_sqrt
from optkit.api import elemwise_inverse_sqrt
from optkit.api import dot
from optkit.api import asum
from optkit.api import nrm2
from optkit.api import axpy
from optkit.api import gemv
from optkit.api import gemm
from optkit.api import cholesky_factor
from optkit.api import cholesky_solve
from optkit.api import splitview
from optkit.api import axpby
from optkit.api import axpby_inplace
from optkit.api import diag
from optkit.api import aidpm
from optkit.api import add_diag
from optkit.api import sum_diag
from optkit.api import norm_diag
from optkit.api import mean_diag
from optkit.api import gramian
from optkit.api import get_curryable_gemv

# Prox calls 
from optkit.api import scale_function_vector
from optkit.api import push_function_vector
from optkit.api import print_function_vector
from optkit.api import func_eval
from optkit.api import prox_eval

# Projector types 
from optkit.api import DirectProjector

# Equilibration methods 
from optkit.api import dense_l2_equilibration
from optkit.api import sinkhornknopp_equilibration

# Solver methods 
from optkit.api import pogs