

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

	for custom types, DO NOT inherit from Adaptor directly, but from a new base class. Then define a new adaptor, for that base class.
	Super super wordy, but the alternative is to register an adaptor against itself, which even I find too confusing.

	EG (this is the pattern taken with the Visitor system):

	# master adaptor class for visiting stuff
	class VisitAdaptor(Adaptor):
		adaptorTypeMap = Adaptor.makeNewTypeMap()

		@classmethod
		def visitObj(obj):
			raise NotImplementedError

	class ListVisitAdaptor(VisitAdaptor):
		# register against builtin list, all good
		forTypes = (list,)

	# a new base class for custom types
	class Visitable: # not inheriting from Adaptor
		def visitThis(self):
			# do whatever instance visit you want
			# DO NOT call or interact with Adaptor here
			pass

	# new adaptor for the custom base class
	class VisitableAdaptor(VisitAdaptor):
		# register against the custom base class
		forTypes = (Visitable,)

		# hook into whatever logic we know will be on the base class
		@classmethod
		def visitObj( cls, obj:Visitable ):
			obj.visitThis()

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
	def checkIsValidAdaptor(cls)->bool:
		"""check if the adaptor is valid
		:raises TypeError if not valid"""
		return True

	@classmethod
	def registerAdaptor(cls, adaptor:type[Adaptor], forTypes:tuple[type, ...]):
		"""Register an adaptor class for a given type.
		"""
		#log("registerAdaptor", adaptor, forTypes)
		#log(adaptor.adaptorTypeMap)
		assert adaptor.adaptorTypeMap, "Must declare empty adaptorTypeMap on Adaptor superclass of {}".format(adaptor)
		assert cls.checkIsValidAdaptor(), f"Adaptor {cls} failed valid check on registration"
		adaptor.adaptorTypeMap.updateClassMap({tuple((set(forTypes))) : adaptor})

	@classmethod
	def adaptorForType(cls, forType:type)->[Adaptor]:
		"""Get the adaptor for the given type.
		"""
		return superClassLookup(cls.adaptorTypeMap.classMap, forType, default=None)

	@classmethod
	def adaptorForObject(cls, forObj:T.Any)->type[Adaptor]:
		"""Get the adaptor for the given object.
		"""
		return cls.adaptorForType(type(forObj))

	@classmethod
	def __init_subclass__(cls, **kwargs):
		"""subclass hook to register against self"""
		super().__init_subclass__(**kwargs)
		if cls.forTypes:
			cls.registerAdaptor(cls, cls.forTypes)
