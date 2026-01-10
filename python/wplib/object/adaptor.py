

from __future__ import annotations
import typing as T
from wplib.inheritance import SuperClassLookupMap

try:
	from wplib.totype import nx
	import networkx as nx
except ImportError:
	print('networkx not installed')
	nx = None

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

	# def __prepare__(self, *args, **kwargs):
	# 	log("adaptor class prepare", args, kwargs)

	adaptorTypeMap : SuperClassLookupMap = None

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
		assert isinstance(cls.adaptorTypeMap, SuperClassLookupMap), "Must declare empty adaptorTypeMap on Adaptor superclass of {}".format(adaptor)
		assert cls.checkIsValidAdaptor(), f"Adaptor {cls} failed valid check on registration"
		adaptor.adaptorTypeMap.updateClassMap({tuple((set(forTypes))) : adaptor})

	@classmethod
	def adaptorForType(cls, forType:type)->[Adaptor]:
		"""Get the adaptor for the given type.
		"""
		#return superClassLookup(cls.adaptorTypeMap.classMap, forType, default=None)
		return cls.adaptorTypeMap.lookup(forType, default=None)

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
			assert isinstance(cls.forTypes, (tuple, set, list)), f"forTypes must be an iterable of types, not {cls.forTypes}"
			cls.registerAdaptor(cls, cls.forTypes)


	# region init shorthand
	"""if dispatchInit is True, adaptor will dispatch a call 
	to its base class, to the type-relevant adaptor subclass
	
	eg 
	class MyAdaptorBase(Adaptor):pass
	class MyListAdaptor(MyAdaptorBase):pass
	
	result = MyAdaptorBase([1,2,3], randomKwarg=1)
	# dispatches to MyListAdaptor([1,2,3], randomKwarg=1)
	"""
	dispatchInit = False

	@staticmethod
	def __new__(cls, *args, **kwargs):
		"""new

		TODO: watch for any jank coming from using __new__ instead of
		__call__ , it's not as safe as I thought

		dispatchInit allows shorthand to get the right adaptor type initialised on the
		argument, eg Pathable([1, 2, 3]) returns a ListPathable wrapping that list

		catch the braindead case where a base class init-dispatches to itself, as a default -
		kwarg __isDispatched denotes the second run of this function
		"""
		#log(f"Adaptor.__new__({cls, args, kwargs})")

		# if this __new__ has dispatched to its own type, just return it
		isDispatched = kwargs.pop("__isDispatched", False)
		if isDispatched:
			return object.__new__(cls)
		#if cls.dispatchInit and not cls.forTypes:
		if cls.__dict__.get("dispatchInit"): # dispatch only if dispatchInit is explicitly declared on this class
			# type coercion if argument is already this adaptor, just returns it
			# excuse me _w_h_a_t_
			# TODO: do not do this
			if isinstance(args[0], cls): return args[0] #this shouldn't be put here at all
			"""opening third eye and trying to have empathy with past ed (difficult),
			I think this was to catch the case of explicitly giving the type to initialise,
			so MySpecialisedAdaptor(args) is guaranteed to return MySpecialisedAdaptor, not
			dynamically dispatch to a sibling type
			but even to do that, it should be issubclass, not isinstance?
			fool 
			"""

			adaptorType = cls.adaptorForType(type(args[0]))
			assert adaptorType, f"No adaptor type {cls} for type {type(args[0])}"
			#return adaptorType(*args, **kwargs)
			return adaptorType.__new__(adaptorType, *args, __isDispatched=True, **kwargs)
		return object.__new__(cls)
		#TODO: figure out how to properly do 'super' here - need to pass it up until it finds
		# a parent class that works differently
		# log("cls", cls, clsSuper(cls))
		# return clsSuper(cls).__new__(cls, )
	# endregion

	# region copying
	"""Adaptor seems like a decent solution to organise per-type behaviour
	and overrides when we can't easily use mixins, but there is an issue - 
	how do we specify per-USECASE overrides, on top of the per-TYPE overrides?
	
	without duplicating the whole thing, and redeclaring the whole thing
	
	SpecialAdaptor = BaseAdaptor.specialise( usage="myUsage", newBases=() )
	
	class SpecialListAdaptor(SpecialAdaptor.adaptorForType(list)):
		# override some stuff
		pass
	???
	
	"""


	# endregion


# TODO:
"""
hitting some cases where Adaptor is
at once too much and too little - 
in general for conversion, we can have an NxN
number of possible conversions between types,
some of which can share logic
none of which can ever conflict (otherwise you shouldn't be using a simple system anyway)
"""


