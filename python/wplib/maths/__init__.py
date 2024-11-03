from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np

"""in future, probably split this off into
a top-level package wpmaths
"""
from wplib.object import Adaptor

class NPArrayLike(Adaptor):
	"""mixin laying out array method template -
	use this as an adaptor for types that can't be extended,
	like Qt containers, MMatrices, etc"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	def __array__(self, dtype=None, copy=None):
		raise NotImplementedError(f"{self}")
	@classmethod
	def fromArray(cls, ar, **kwargs):
		raise NotImplementedError(f"{cls}")
NPArrayLike.forTypes = (NPArrayLike, )

arrT = np.ndarray
def arr(obj, dtype=None, copy=None):
	"""all-purpose function to get a numpy array for
	the given object, invoking the ArrayLike adaptor
	if needed
	"""
	adaptor = NPArrayLike.adaptorForObject(obj)
	if adaptor is not None: return adaptor.__array__(obj, dtype, copy)
	return np.array(obj)

def fromArr(ar, cls, **kwargs):
	"""return an instance of cls from the given array
	a bit annoying to have to call this, and not have it
	implicit in init or something
	"""
	if isinstance(ar, cls): return ar # it's already the right kind of object
	adaptor = NPArrayLike.adaptorForType(cls)
	if adaptor is not None: return adaptor.fromArray(ar, **kwargs)
	return cls(ar)


