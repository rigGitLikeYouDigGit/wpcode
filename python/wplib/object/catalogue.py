from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


"""super simple mixin to index subclasses against
a constant value.
TODO: combine with Adaptor
TODO: or maybe not
"""

EVAL = lambda x : x() if isinstance(x, (types.FunctionType, types.MethodType)) else x

class Catalogue:
	catalogue = {} # redeclare on new base classes for separate maps
	if T.TYPE_CHECKING:
		catalogueKey = "" # string or callable for the index key for this class
	def __init_subclass__(cls, **kwargs):
		if hasattr(cls, "catalogueKey"):
			cls.catalogue[EVAL(cls.catalogueKey)] = cls
		else:
			cls.catalogue[cls.__name__] = cls
	@classmethod
	def getCatalogueCls(cls, key):
		return cls.catalogue.get(key)
