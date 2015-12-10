# CPU32, CPU64, GPU32, GPU64
class OKBackend(object):
	def __init__(self, GPU=False, double_precision=True):
		self.current_precision = '64-bit' if double_precision else '32-bit'
		self.current_device = 'CPU' if not GPU else 'GPU'
		self.dense = None
		self.sparse = None
		self.prox = None

	def change(GPU=False, double_precision=True):
		self.current_precision = '64-bit' if double_precision else '32-bit'
		self.current_device = 'CPU' if not GPU else 'GPU'
		self.dense = None
		self.sparse = None
		self.prox = None

Backend = OKBackend()


