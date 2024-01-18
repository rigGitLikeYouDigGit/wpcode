

from __future__ import annotations
import typing as T

from wplib.log import log
from wplib.inheritance import SuperClassLookupMap, superClassLookup, clsSuper

class Adaptor:
	"""Abstract-abstract class for registering "companion" objects
	to types, for type-specific behaviour without subclassing.

	Adaptors can also be base class mixins for custom types (in this case
	the adaptor is registered against itself)

	For example, a VisitableAdaptor base class might define visitor behaviour
	for a builtin list, or dataclass, or numpy array etc

	for custom types, would this mean both a custom lightweight base class AND
	an external adaptor for it?

	use 2 parts : a base Adaptor, and a base class mixin for custom types
	"""

	adaptorTypeMap : SuperClassLookupMap = None

	# reflexive = True # if true, adaptor is registered against itself
	# # allows inheriting from adaptor directly into active objects
	# testing full separation

	@classmethod
	def makeNewTypeMap(cls):
		"""make a new type map for this class"""
		return SuperClassLookupMap({})

	# if T.TYPE_CHECKING:
	forTypes : tuple[type, ...] = () # no active impact, just helps type hinting

	@classmethod
	def registerAdaptor(cls, adaptor:type[Adaptor], forTypes:tuple[type, ...]):
		"""Register an adaptor class for a given type.
		"""
		#log("registerAdaptor", adaptor, forTypes)
		#log(adaptor.adaptorTypeMap)
		assert adaptor.adaptorTypeMap, "Must declare empty adaptorTypeMap on Adaptor superclass of {}".format(adaptor)
		adaptor.adaptorTypeMap.updateClassMap({forTypes : adaptor})

	@classmethod
	def adaptorForType(cls, forType:type)->Adaptor:
		"""Get the adaptor for the given type.
		"""
		return superClassLookup(cls.adaptorTypeMap.classMap, forType, default=None)

	@classmethod
	def __init_subclass__(cls, **kwargs):
		"""subclass hook to register against self"""
		super().__init_subclass__(**kwargs)
		if cls.forTypes:
			cls.registerAdaptor(cls, cls.forTypes)
