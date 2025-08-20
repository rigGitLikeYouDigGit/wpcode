
from __future__ import annotations
import typing as T

import functools, inspect, builtins
from functools import partial, partialmethod

from wplib import log

from wplib.object import UserDecorator

"""TODO: refactor to use To, move this into To actually
"""

"""decorator to automatically convert arguments to a single
annotated type"""


class coerce(UserDecorator):
	"""Where annotations are given for function's arguments,
	coerce incoming objects to them

	@coerce
	def fn(a:int, b:str, c, d:(float, str):
		...

	a and b will be coerced;
	c has no given annotation;
	d has multiple, so skip it (for now)

	"""

	if T.TYPE_CHECKING:
		def __init__(self, typeConvertFnMap:dict[str, callable]=None):
			"""typeConvertFnMap is a dict of type names to functions
			which can convert to that type
			"""

	# def __init__(self, *args, typeConvertFnMap:dict[str, callable]=None, **kwargs):
	# 	"""typeConvertFnMap is a dict of type names to functions
	# 	which can convert to that type
	# 	"""
	# 	self.typeConvertFnMap = typeConvertFnMap or {}
	# 	super().__init__(*args, typeConvertFnMap=typeConvertFnMap, **kwargs)

	@classmethod
	def getAvailableTypeNames(cls)->dict[str, type]:
		"""return dict of all available type names"""
		return {k:v for k, v in globals().items() if isinstance(v, type)}

	#
	def wrapFunction(self,
	                 targetFunction:callable,
	                 decoratorArgsKwargs:(None, tuple[tuple, dict])=None) ->function:
		"""OVERRIDE
		convert incoming arguments to annotated types

		annotation names MUST be imported as full types
		where function is defined, we're not doing any
		crazy deferred import here
		"""
		log("wrapFunction", targetFunction, decoratorArgsKwargs)

		self.typeConvertFnMap = self.kwargs.get("typeConvertFnMap", {})

		# get function annotations
		anns = targetFunction.__annotations__
		log("anns", anns)

		# get globals of calling module
		outerFrame = inspect.currentframe().f_back
		outerGlobals = dict(outerFrame.f_globals)
		outerGlobals.update(builtins.__dict__)
		outerGlobals.pop("copyright")
		outerGlobals.pop("__doc__")
		outerGlobals.pop("credits")

		# dict of argument name to target type
		argNameTypeMap = {}

		for argName, argTypeStr in anns.items():
			try:
				# argtypestr is just a string, so we can't catch tuples
				# has to be eval here to catch tuples of types
				argType = eval(argTypeStr, outerGlobals)
			except KeyError:
				raise TypeError(
					f"Unable to find type name '{argTypeStr}' in calling module's globals:\n{outerGlobals};\nCannot coerce arguments {argName} of function {targetFunction.__name__}"
				)
			# check for tuples, skip if found
			if not isinstance(argType, type):
				log("skipping", argType)
				continue
			argNameTypeMap[argName] = argType


		@functools.wraps(targetFunction)
		def wrapper(*args, **kwargs):
			bound = inspect.signature(targetFunction).bind(*args, **kwargs)
			for k, v in bound.arguments.items():

				# if no valid annotation, skip
				if k not in argNameTypeMap:
					continue

				# if already correct type, skip
				if isinstance(v, argNameTypeMap[k]):
					continue

				# if in type convert map, use that
				if k in self.typeConvertFnMap:
					try:
						bound.arguments[k] = self.typeConvertFnMap[k](v)
						continue
					except Exception as e:
						e.args = (*e.args, (
							f"Cannot coerce argument {v} of param {k} to type {argNameTypeMap[k]} for function {targetFunction.__name__};\n tried to use typeConvertFnMap {self.typeConvertFnMap} function {self.typeConvertFnMap[k]} "
						))
						raise e

				# convert directly
				try:
					# convert to type
					bound.arguments[k] = argNameTypeMap[k](v)
				except Exception as e:
					e.args = (*e.args, (
						f"Cannot directly coerce argument {v} of param {k} to type {argNameTypeMap[k]} for function {targetFunction.__name__}"
					))
					raise e
			return targetFunction(**bound.arguments)
		return wrapper



if __name__ == '__main__':

	@coerce
	def printArgTypes(
			a:int, b:int, c, d:(float, str), e, f:str="default"
	):
		for i in locals().items():
			print(i, type(i[1]))



	printArgTypes(1, "2", 3, 4.0, 5.0, "6")

	# following raises coercion error
	#printArgTypes(1, ["2"], 3, 4.0, 5.0, "6")


