
from __future__ import annotations
"""tiny base class for object references - looking
up disparate objects dynamically based on given settings

to be used where actual objects may be created or destroyed frequently

may also be used for consistent handles to code elements within a changing
codebase, somehow

consider that it may be desired to track objects by a single invariate
uid, or by namespace or path - for the purposes of changing the result of serialised references, etc

stick with uids for now.


"""
from importlib import import_module


class ObjectReference:
	"""object should have is logic defined by its parametres -
	actual resolve() method should only put that logic to use,
	not accept any logic-altering arguments
	"""
	# type of object to return from resolving
	valueType : object

	def resolve(self)->valueType:
		raise NotImplementedError

	def __copy__(self):
		"""used to get reference to the same object"""
		raise NotImplementedError

"""if given is reference, resolve it, else return it"""
resolve = lambda x: x.resolve() if isinstance(x, ObjectReference) else x


class TypeReference(ObjectReference):
	"""reference to a type object - should be as robust as possible
	to those objects changing names, changing places etc
	OR we can just use names, and selectively use separate tools
	to update any saved references"""
	valueType = type

	serialiseDataKey = "@_typeRef"

	@staticmethod
	def moduleFromClass(cls):
		return cls.__module__

	def __init__(self, typeToReference:type=None, typeName="", typeModule=""):
		self._typeToReference = None
		if typeToReference is not None:
			self._typeToReference = typeToReference
			self._typeName = typeToReference.__name__
			self._moduleName = self.moduleFromClass(typeToReference)
		else:
			self._typeName = typeName
			self._moduleName = typeModule

	def resolve(self)->type:
		"""look up type from saved name and module """
		# return self._typeToReference
		module = import_module(self._moduleName)
		return getattr(module, self._typeName)

	def serialise(self)->tuple:
		return (self._typeName, self._moduleName)

	@classmethod
	def deserialise(cls, dataTuple)->cls:
		return cls(typeToReference=None,
		           typeName=dataTuple[0],
		           typeModule=dataTuple[1])

	def __copy__(self):
		"""used to get reference to the same object"""
		return TypeReference(typeToReference=None,
		                     typeName=self._typeName,
		                     typeModule=self._moduleName)

