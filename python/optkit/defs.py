from numpy import float32, float64

__float_flag=False
__sparse_flag=False
__gpu_flag=False

__float_tag='32' if __float_flag else '64'
__sparse_tag='sparse' if __sparse_flag else 'dense'
__gpu_tag='gpu' if __gpu_flag else 'cpu'

__float_conversion = float32 if __float_flag else float64