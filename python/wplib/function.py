from __future__ import annotations

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