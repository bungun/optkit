import unittest

class OptkitCTestCase(unittest.TestCase):
	managed_vars = {}
	free_methods = {}

	def register_var(self, name, var, free):
		self.managed_vars[name] = var;
		self.free_methods[name] = free;

	def unregister_var(self, name):
		self.managed_vars.pop(name, None)
		self.free_methods.pop(name, None)

	def free_var(self, name):
		var = self.managed_vars.pop(name, None)
		free_method = self.free_methods.pop(name, None)
		if free_method is not None and var is not None:
			free_method(var)

	def free_all_vars(self):
		for varname in self.managed_vars.keys():
			print 'releasing unfreed C variable {}'.format(varname)
			self.free_var(varname)