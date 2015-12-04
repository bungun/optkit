from numpy import float32, float64
import os

FLOAT_FLAG=False
SPARSE_FLAG=False
GPU_FLAG=False
DIMCHECK_FLAG=True
TYPECHECK_FLAG=True

FLOAT_TAG='32' if FLOAT_FLAG else '64'
SPARSE_TAG='sparse' if SPARSE_FLAG else 'dense'
GPU_TAG='gpu' if GPU_FLAG else 'cpu'

FLOAT_CAST = float32 if FLOAT_FLAG else float64

MACHINETOL = 1e-5 if FLOAT_FLAG else 1e-10



if os.uname()[0] == "Darwin":
	OK_HOME = '/Users/Baris/Documents/Thesis/modules/optkit/'
else:
	OK_HOME = '/home/baris/optkit/'	
