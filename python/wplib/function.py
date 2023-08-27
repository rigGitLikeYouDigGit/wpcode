from __future__ import annotations

import ast
import inspect

from types import FunctionType


def addToFunctionGlobals(fn:FunctionType, customGlobals:dict):
	"""add custom globals to function, without affecting the base builtins"""
	#### W A R N I N G ####
	# extremely cursed python ahead #

	baseBuiltins = fn.__globals__["__builtins__"] # Schroedinger's variable
	# baseBuiltins is just a normal dict here if you don't do anything to it
	# UNLESS you try to set a value, LATER IN THE CODE
	# effects of this reach back in time, and turn baseBuiltins above into a module object

	# try it yourself, comment out the set line below and check the type of baseBuiltins

	#print(type(baseBuiltins))
	#print(baseBuiltins.keys(), type(baseBuiltins))

	try:
		baseBuiltins.update(customGlobals) # this line will fail
		# the presence of this line, itself causes baseBuiltins above to be a module
		# by existing, it undoes itself
		# I can relate

	except (AttributeError, TypeError): # which will always be raised
		baseBuiltins.__dict__.update(customGlobals) # this line will succeed
		fn.__globals__["__builtins__"] = baseBuiltins.__dict__

if __name__ == '__main__':

	def fn(a, b, *args, c=3, **kwargs):
		"""test function"""
		print(a, b, args, c, kwargs)

	spec = inspect.signature(fn)
	print(spec)
	for i in spec.parameters.values():
		print(i.__reduce__())
		print(i.kind, i.name, i.default, i.annotation)
	#print(spec.parameters)


