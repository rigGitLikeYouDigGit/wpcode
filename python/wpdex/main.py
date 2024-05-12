
from __future__ import annotations
import typing as T


from wplib import log
from wplib.object import Adaptor, TypeNamespace, HashIdElement
from wplib.object.visitor import VisitAdaptor, Visitable

class DexPathable:
	"""base class for objects that can be pathed to"""
	keyT = T.Union[str, int]
	keyT = T.Union[keyT, tuple[keyT, ...]]
	pathT = T.List[keyT]

	# region combine operators
	class Combine(TypeNamespace):
		"""operators to flatten multiple results into one"""
		class _Base(TypeNamespace.base()):
			@classmethod
			def flatten(cls, results:(list[T.Any], list[DexPathable]))->T.Any:
				"""flatten results"""
				raise NotImplementedError(cls, f"no flatten for {cls}")

		class First(_Base):
			"""return the first result"""
			@classmethod
			def flatten(cls, results:(list[T.Any], list[DexPathable]))->T.Any:
				"""flatten results"""
				return results[0]
	# endregion

	def path(self)->pathT:
		"""return path to this object"""
		raise NotImplementedError(self)

class DexValidator:
	"""base class for validating dex objects"""
	def validate(self):
		"""validate this object"""
		raise NotImplementedError(self)


class WpDex(Adaptor,  # integrate with type adaptor system
            HashIdElement,  # each wrapper has a transient hash for caching

            # interfaces that must be implemented
            Visitable,  # compatible with visitor pattern
            DexPathable, DexValidator,
            ):
	"""base for wrapping arb structure in a
	WPDex graph, allowing pathing, metadata, UI generation,
	validation, etc

	Core data in WPDex structure is IMMUTABLE - whenever anything changes,
	WPDex is regenerated. Detecting these changes from the outside is
	another issue

	Regardless of how we structure it, there's going to be a lot of
	stuff here. Do we prefer a few large blocks of stuff, or many small pieces
	of stuff

	Could split up classes per-type, per-behaviour - DictDexVisitable, ListDexValidator, etc

	let's just have single objects and interfaces - DictDex, ListDex, etc

	THIS base class collects together the interfaces that need to be fulfilled, and subclasses
	each implement them differently.
	This actually doesn't seem too bad


	for defining different behaviour in different regions of structure, different areas
	of program, etc,
	use paths to set overrides rather than defining new classes
	"""
	# adaptor integration
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	dispatchInit = True # WPDex([1, 2, 3]) will return a specialised ListDex object

	def __init__(self, obj:T.Any, parent:DexPathable=None, key:T.Iterable[DexPathable.keyT]=None):
		"""initialise with object and parent"""
		# superclass inits
		HashIdElement.__init__(self)
		#Adaptor.__init__(self)

		# should these attributes be moved into pathable? probably
		self.parent = parent
		self.obj = obj
		self.key = list(key or [])


class DictDex(WpDex):
	"""dict dex"""
	forTypes = (dict,)

	def path(self) ->pathT:
		"""return path to this object"""
		return []

	def validate(self):
		"""validate this object"""
		pass




