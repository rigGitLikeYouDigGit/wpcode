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
from . import arrT

def indexValueArrsFromTupleList(
		tupleList:list[tuple[int | float, ...]],
)->tuple[np.ndarray, np.ndarray]:
	"""return index arr of N+1 entries, starting at 0
	and flat value array"""
	n = len(tupleList)
	indices = np.empty(n + 1, dtype=int)
	indices[0] = 0

	# calculate total size and fill indices
	total_size = 0
	for i, tup in enumerate(tupleList):
		total_size += len(tup)
		indices[i + 1] = total_size

	# pre-allocate values array and fill
	values = np.empty(total_size, dtype=float)
	for i, tup in enumerate(tupleList):
		values[indices[i]:indices[i + 1]] = tup

	return indices, values

def tupleListFromIndexValueArrs(
		indices:np.ndarray,
		values:np.ndarray,
)->list[tuple[int | float, ...]]:
	"""reconstruct tuple list from index and value arrays"""
	n = len(indices) - 1
	result = [()] * n
	for i in range(n):
		result[i] = values[indices[i] : indices[i + 1]]
	return result


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

def v3(*args)->arrT:
	vals = args[0] if isinstance(args[0], (tuple, list, np.ndarray)) else args[:3]
	return np.array(vals, dtype=float)
def segmentFromPoints(*args)->arrT:
	"""put given arguments in np array as entries"""
	i = 0
	vals = []
	while i < len(args):
		if not isinstance(args[i], (tuple, list, np.ndarray)): break
		vals.append(args[i])
		i += 1
	return np.array(vals, dtype=float)
#lineFromPoints = line
def segmentFromDir(dir:v3,
				origin:v3=(0, 0, 0),
                t0:float=0,
                t1:float=1,
                *args
                )->arrT:
	"""try to find the most fluid, ergonomic
	order of arguments here"""
	return np.array([origin + dir * t0,
	             origin + dir * t1])

def infLineFromDir(dir:v3,
                   origin:v3=np.array((0, 0, 0))):
	"""convert to form [a, b, c, d]
	for aX + bY + cZ = d"""
	return np.array([dir, origin])

def distanceFromInfLine(segment,
                        pos:v3=np.array((0, 0, 0))):
	"""expect line as [pt1, pt2]
	to a point p0 (from wolframAlpha)"""
	return np.linalg.norm( np.cross(pos - segment[0], pos - segment[1])) / \
			np.linalg.norm( segment[2] - segment[1] )


def posOnSegment(segment:arrT,
                   t:float=0.0):
	return (1.0 - t) * segment[0] + t * segment[1]

def closestPosOnSegment(segment:arrT,
                        pos:v3=np.array((0, 0, 0))):
	#nul = np.zeros(3)
	v = segment[1] - segment[0]
	u = segment[0] - pos
	t = - np.dot(v, u) / np.dot(v, v)
	if (0.0 <= t) and ( t <= 1.0):
		return posOnSegment(segment, t)
	return segment[int(np.linalg.norm(pos - segment[0]) >
	                   np.linalg.norm(pos - segment[1]))]

def closestParamOnSegment(segment:arrT,
                        pos:v3=np.array((0, 0, 0))):
	#nul = np.zeros(3)
	v = segment[1] - segment[0]
	u = segment[0] - pos
	t = - np.dot(v, u) / np.dot(v, v)
	if (0.0 <= t) and ( t <= 1.0):
		return t
	return int(np.linalg.norm(pos - segment[0]) >
	                   np.linalg.norm(pos - segment[1]))


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