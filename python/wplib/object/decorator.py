
from __future__ import annotations

import pprint
from functools import wraps, partial
import typing as T


class UserDecorator:
	"""
	small class to make it easier to define
decorators -
formalises accepting no arguments, an empty call, or arbitrary arguments

	@decorator
	def func():
	calls wrapFunction() with no decoratorArgsKwargs

	also provides uniform interface for functions vs methods

	for now not providing separate method for "callFunction()" - still
	need to return the outer wrapper from wrapFunction()

	this way we keep some more familiar syntax from normal
	decorator declaration
	"""

	def __init__(self, *args, **kwargs):
		#print("init args", args, kwargs)

		self._targetFunction = None
		self._wrappedFunction = None

		self.args = ()
		self.kwargs = {}

		# bound instance may be populated if this decorator class is
		# used as descriptor within another class
		self.boundInstance = None

		if len(args) == 1 and callable(args[0]): # no arguments given
			self._targetFunction = args[0]
			self.args = args[1:]
			self.kwargs = kwargs
			if self.args or self.kwargs:
				decoArgsKwargs = (self.args, self.kwargs)
			else:
				decoArgsKwargs = None
			self._wrappedFunction = self.wrapFunction(
				self._targetFunction, decoratorArgsKwargs=decoArgsKwargs)
		else:
			self.args = args
			self.kwargs = kwargs

	def __get__(self, instance, owner):
		"""defining a descriptor method like this affects how the class
		binds to instances and instance methods, I don't fully understand it
		"""
		# populate own bound instance attribute
		self.boundInstance = instance
		#print("UD _get_", owner)
		# call this descriptor with the bound instance
		return partial(self.__call__, instance)

	def __call__(self, *args, **kwargs):
		#print("call args", args, kwargs)
		#print("call descriptor", self, "bound", self.boundInstance, args, kwargs)
		#print("call")

		if self._wrappedFunction: # already wrapped, no decorator arguments given
			# return called wrapped function
			return self._wrappedFunction(*args, **kwargs)

		# instead the function is given here
		self._targetFunction = args[0]
		if self._targetFunction is self.boundInstance:
			args = args[1:]
			self._targetFunction = args[0]
		self._wrappedFunction = self.wrapFunction(
			self._targetFunction, (self.args, self.kwargs)
		)
		return self._wrappedFunction

	def wrapFunction(self,
	                 targetFunction:callable,
	                 decoratorArgsKwargs:(None, tuple[tuple, dict])=None)->function:
		"""OVERRIDE

		if decoratorArgsKwargs is None,
		decorator was called without arguments
		else, a tuple of (args, kwargs)

		return the direct function to call - usually
		the new wrapper function
		"""
		return targetFunction



if __name__ == '__main__':

	@UserDecorator
	def a():
		pass

	print(a)
	a()

	print("")
	@UserDecorator("arg")
	def b():
		pass

	print(b)
