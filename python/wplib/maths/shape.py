from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import numpy as np

"""turns out you 
can override every common np array function with custom classes - 
for now we only do top-level stuff and convenience inits

SHOULD we do purely functional? muddies the water if we 
can guarantee subclasses in some places, and some not
"""


class _RealisticInfoArray(np.ndarray):
	"""copied from np docs as reference"""

	def __new__(cls, input_array, info=None):
		# Input array is an already formed ndarray instance
		# We first cast to be our class type
		obj = np.asarray(input_array).view(cls)
		# add the new attribute to the created instance
		obj.info = info
		# Finally, we must return the newly created object:
		return obj

	def __array_finalize__(self, obj):
		# see InfoArray.__array_finalize__ for comments
		if obj is None: return
		self.info = getattr(obj, 'info', None)


class ArrayLike(np.ndarray):
	"""barely worth a base class"""

class V3(np.ndarray):
	"""just copy-paste the __new__ code for each of these"""
	shape = (3, )

	def __new__(cls, *args, **kwargs):
		vals = args[0] if isinstance(args[0], (tuple, list, np.ndarray)) else args[:3]
		obj = np.asarray(np.array(vals, dtype=float)).view(cls)
		return obj

class Line(np.ndarray):
	shape = (2, 3)
	@classmethod
	def _getValsFromNew(cls, args, kwargs):
		i = 0
		vals = []
		while i < len(args):
			if not isinstance(args[i], (tuple, list, np.ndarray)): break
			vals.append(args[i])
			i += 1
		return vals

	def __new__(cls, arr=None, *args, **kwargs):
		vals = cls._getValsFromNew(args, kwargs)
		assert len(vals) == cls.shape[0]
		obj = np.asarray(np.array(vals, dtype=float)).view(cls)
		return obj

def v3(*args):
	vals = args[0] if isinstance(args[0], (tuple, list, np.ndarray)) else args[:3]
	return np.array(vals, dtype=float)
def line(*args):
	i = 0
	vals = []
	while i < len(args):
		if not isinstance(args[i], (tuple, list, np.ndarray)): break
		vals.append(args[i])
		i += 1
	return np.array(vals, dtype=float)
lineFromPoints = line
def lineFromDir(dir:v3,
				origin:v3=(0, 0, 0),
                t0:float=0,
                t1:float=1,
                *args
                ):
	"""try to find the most fluid, ergonomic
	order of arguments here"""
	dir = np.array(dir)
	origin = np.array(origin)
	return np.array([origin + dir * t0,
	             origin + dir * t1])


# lineFromDir((1, 1, 0),
#             origin=())



if __name__ == '__main__':

	v = V3(1, 2, 3)
	log("v", v, type(v), v + 3.0)

	# raw = [(1, 2, 3),
	# 		  (2, 2, 2)]
	# l = Line(raw)
	# a = np.array(raw)
	# a.at = "ey"
	# log(a.at)

	pass