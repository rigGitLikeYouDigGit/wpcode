from __future__ import annotations, print_function

import typing as T


def leafParentBases(*desiredBases:tuple[type])->list[type]:
	"""given selection of superclasses to inherit from,
	return the actual bases to pass, to generate working mro

	eg if one is a subclass of the other, only pass the lowest base

	super temp for now, come back and make this work properly if needed
	"""
	#mainSeq = set(desiredBases[0].__mro__)
	resultBases = {desiredBases[0]}
	for secondaryBase in desiredBases[1:]:
		resultBases -= set(secondaryBase.__mro__)
		resultBases.add(secondaryBase.__mro__[0])
	return resultBases


def containsSuperClass(classSeq:T.Sequence[type], lookup:(type, object))->(type, None):
	"""returns any items in the sequence which are superclasses of lookup"""
	if not isinstance(lookup, type):
		lookup = type(lookup)
	for i in classSeq:
		#if i in lookup.__mro__:
		if issubclass(lookup, i): # really hope this works
			return i
	return None

def superClassLookup(classMap:(dict[type], dict[tuple[type]]), lookupCls:(type, object), default=None):
	"""indexes in to a map of {type : value} using lookupCls"""
	matching = containsSuperClass(classMap, lookupCls)
	if matching is None: return default
	return classMap[matching]


def iterSubClasses(cls, _seen=None, includeTopCls=False)->T.Generator[T.Type["cls"]]:
	"""
	iterSubClasses(cls)
	http://code.activestate.com/recipes/576949-find-all-subclasses-of-a-given-class/
	Generator over all subclasses of a given class, in depth first order.
	"""

	if not isinstance(cls, type):
		raise TypeError('iterSubClasses must be called with '
						'new-style classesToReload, not %.100r' % cls)
	if _seen is None: _seen = set()
	try:
		subs = cls.__subclasses__()
	except TypeError:  # fails only when cls is type
		subs = cls.__subclasses__(cls)
	if includeTopCls:
		subs = [cls] + list(subs)
	for sub in subs:
		if sub not in _seen:
			_seen.add(sub)
			yield sub
			for sub in iterSubClasses(sub, _seen, includeTopCls=False):
				yield sub

def mroMergedDict(cls):
	"""returns a merged dict of all superclasses of cls,
	matching override order during inheritance"""
	merged = {}
	for i in cls.__mro__:
		merged.update(i.__dict__)
	return merged

# annotation decorators
def overrideThis(fn:T.Callable)->T.Callable:
	"""decorator to mark a method as needing to be overridden
	"""
	return fn