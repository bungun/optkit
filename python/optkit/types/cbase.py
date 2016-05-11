class OptkitBaseTypeC(object):
	__exit_default = lambda arg: None

	def __init__(self, backend):
		self.__exit_call = self.__exit_default
		self.__exit_arg = None
		self.__backend = backend
		self.__reset_on_exit = False
		self.__registered = False

	def __del__(self):
		if self.registered:
			self.__unregister_c()
		args = [self.exit_arg]
		if self.reset_on_exit:
			args.append(int(self.backend.device_reset_allowed))
		self.__exit_call(*args)

	@property
	def registered(self):
		return self.__registered

	def __register_c(self):
		self.backend.increment_cobject_count()
		self.__registered = True

	def __unregister_c(self):
		self.backend.decrement_cobject_count()
		self.__registered = False

	@property
	def reset_on_exit(self):
	    return self.__reset_on_exit

	@reset_on_exit.setter
	def reset_on_exit(self, reset):
		self.__reset_on_exit = bool(reset)

	@property
	def exit_arg(self):
		return self.__exit_arg

	@exit_arg.setter
	def exit_arg(self, exit_arg):
		self.__exit_arg = exit_arg