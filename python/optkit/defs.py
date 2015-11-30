from numpy import float32, float64

FLOAT_FLAG=False
SPARSE_FLAG=False
GPU_FLAG=False
DIMCHECK_FLAG=True
TYPECHECK_FLAG=True

FLOAT_TAG='32' if FLOAT_FLAG else '64'
SPARSE_TAG='sparse' if SPARSE_FLAG else 'dense'
GPU_TAG='gpu' if GPU_FLAG else 'cpu'

FLOAT_CAST = float32 if FLOAT_FLAG else float64

OK_HOME = '/home/baris/optkit/'
