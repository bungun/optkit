from numpy import zeros

class PogsVariablesLocal():
	def __init__(self, m, n, pytype):
		self.m = m
		self.n = n
		self.z = zeros(m + n).astype(pytype)
		self.z12 = zeros(m + n).astype(pytype)
		self.zt = zeros(m + n).astype(pytype)
		self.zt12 = zeros(m + n).astype(pytype)
		self.prev = zeros(m + n).astype(pytype)
		self.d = zeros(m).astype(pytype)
		self.e = zeros(n).astype(pytype)

	@property
	def x(self):
		return self.z[self.m:]

	@property
	def y(self):
		return self.z[:self.m]

	@property
	def x12(self):
		return self.z12[self.m:]

	@property
	def y12(self):
		return self.z12[:self.m]

	@property
	def xt(self):
		return self.zt[self.m:]

	@property
	def yt(self):
		return self.zt[:self.m]

	@property
	def xt12(self):
		return self.zt12[self.m:]

	@property
	def yt12(self):
		return self.zt12[:self.m]

class PogsOutputLocal():
	def __init__(self, denselib, pogslib, m, n):
		self.x = zeros(n).astype(denselib.pyfloat)
		self.y = zeros(m).astype(denselib.pyfloat)
		self.mu = zeros(n).astype(denselib.pyfloat)
		self.nu = zeros(m).astype(denselib.pyfloat)
		self.ptr = pogslib.pogs_output(
				self.x.ctypes.data_as(denselib.ok_float_p),
				self.y.ctypes.data_as(denselib.ok_float_p),
				self.mu.ctypes.data_as(denselib.ok_float_p),
				self.nu.ctypes.data_as(denselib.ok_float_p))