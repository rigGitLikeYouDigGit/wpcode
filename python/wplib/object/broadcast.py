from __future__ import annotations
import types, typing as T
import pprint
from wplib import log


from wplib.object.adaptor import Adaptor

"""
I really haven't been the same since the first time I touched a numpy array

Desire here is to add Numpy-esque broadcasting logic into 
custom types, where useful
"""



class BroadcastAdaptor(Adaptor):
	"""adaptor defining logic for
	expanding types in place, during
	broadcasting operations

	It might be unlikely you would ever need to extend
	these functions for unpacking/unrolling custom types beyond Maya,
	but it's happened at least once, and this way also helps me reason about it

	ACTUALLY I think adaptors are too universal - logic of broadcasting in different
	contexts is probably specific enough that we would subclass the Broadcaster anyway?
	With adaptors we assume logic is the same across all types ever declared -
	seems too complicated and too scattered
	"""
	adaptorTypeMap = Adaptor.makeNewTypeMap()
	forTypes = (object, )

	@classmethod
	def _getElements(cls, obj):
		"""return EXPANDABLE version of obj?
		no, can't bake in 'only one level'
		of immutability
		or maybe we can, maybe that's the most sane thing to do
		"""
		# if isinstance(obj, PlugBase):
		# 	obj = obj.MPlug
		# if isinstance(obj, om.MPlug):
		# 	return _getElementsMPlug(obj)

		if isinstance(obj, (tuple, list)):
			return list(obj)
		return [obj]

	@classmethod
	def _isLeaf(cls, obj): #TODO: make superclass lookup maps for _isLeaf and _getElements
		# if isinstance(obj, PlugBase):
		# 	obj = obj.MPlug
		# if isinstance(obj, om.MPlug):
		# 	if obj.isElement:
		# 		return not (obj.isArray or obj.isCompound or obj.parent().isCompound)
		# 	return not (obj.isArray or obj.isCompound)
		if isinstance(obj, (tuple, list)):
			return False
		return True

	@classmethod
	def _complexSourceMatchesLeafTarget(cls, possible, thisLeaf):
		"""MAYBE???
		for the case of floatArray plugs, where a normally non-leaf value
		might match a specific leaf target

		called on the adaptor matching the type OF THE LEAF.
		"""
		return False

	@classmethod
	def _complexSidesMatchDirect(cls, src, thisDst):
		""" check if 2 arbitrary complex objects
		can easily be said to match -
		EG if you have 2 compound attributes with the same structure.

		in this case, this pair is yielded and no more recursion done
		called on the adaptor matching type OF THE DESTINATION.
		"""
		return False


class Broadcaster:
	"""object giving logic for expanding / truncating / broadcasting
	values into a destination structure,
	similar to Numpy arrays

	is it really worth factoring this?
	"""


	def _getElements(self, obj):
		"""return EXPANDABLE version of obj?
		no, can't bake in 'only one level'
		of immutability
		or maybe we can, maybe that's the most sane thing to do
		"""
		# if isinstance(obj, PlugBase):
		# 	obj = obj.MPlug
		# if isinstance(obj, om.MPlug):
		# 	return _getElementsMPlug(obj)

		if isinstance(obj, (tuple, list)):
			return list(obj)
		return [obj]

	def _isLeaf(self, obj): #TODO: make superclass lookup maps for _isLeaf and _getElements
		# if isinstance(obj, PlugBase):
		# 	obj = obj.MPlug
		# if isinstance(obj, om.MPlug):
		# 	if obj.isElement:
		# 		return not (obj.isArray or obj.isCompound or obj.parent().isCompound)
		# 	return not (obj.isArray or obj.isCompound)
		if isinstance(obj, (tuple, list)):
			return False
		return True

	def _complexSourceMatchesLeafTarget(self, possible, thisLeaf):
		"""MAYBE???
		for the case of floatArray plugs, where a normally non-leaf value
		might match a specific leaf target

		"""
		return False

	def _complexSidesMatchDirect(self, src, thisDst):
		""" check if 2 arbitrary complex objects
		can easily be said to match -
		EG if you have 2 compound attributes with the same structure.

		in this case, this pair is yielded and no more recursion done
		"""
		return False

	def _isImmutable(self, obj):
		return isinstance(obj, (str, tuple, float, int))#, om.MMatrix, om.MPlug))

	def broadcast(self, a, b):
		""" expand 2 inputs to a list of matched pairs
		EITHER both are leaf
		OR neither is leaf
		OR left is leaf
		OR right is leaf

		not trying to check for depth yet, goes level-by-level from both roots in step


		B is destination - structure to match

		A- find a way to extend immutable entries
		"""
		# log("broadcast", a, b, _isLeaf(a), _isLeaf(b))
		isLeafA = self._isLeaf(a)
		isLeafB = self._isLeaf(b)
		if isLeafA and isLeafB:
			yield (a, b)
			return

		# if target is a leaf, check if complex source matches it
		if isLeafB:
			if self._complexSourceMatchesLeafTarget(a, b):
				yield (a, b)
				return
			# over-truncating is an error more than a help
			raise RuntimeError("Tried to broadcast non-leaf source {} to leaf target {}".format(a, b))

		# if source is a leaf, expand out targets
		if isLeafA:
			for t in self._getElements(b):
				yield from self.broadcast(a, t)
			return

		# neither source nor dest is a leaf, it gets complicated
		# check for direct match:
		if self._complexSidesMatchDirect(a, b):
			yield a, b
			return

		targets = self._getElements(b)
		if self._isImmutable(a):
			# sources = [a] * len(targets)
			sources = [self._getElements(a)] * len(targets)
			for left, right in zip(sources, targets):
				yield from self.broadcast(left, right)
			return

		# truncate to shortest
		sources = self._getElements(a)
		# truncate to shortest
		shortestLen = min(len(sources), len(targets))

		sources = sources[:shortestLen]
		targets = targets[:shortestLen]
		for src, dst in zip(sources, targets):
			yield from self.broadcast(src, dst)


# def _getElementsSeq(obj:(tuple, list)):
# 	return obj
#
# def _getElementsMPlug(obj:om.MPlug):
# 	if obj.isCompound:
# 		return tuple(obj.child(i) for i in range(obj.numChildren()))
# 	if obj.isArray: # absolutely no damn idea
# 		raise RuntimeError("nooooooooooooo")
# 	return [obj]
#
#
#
# def _getElements(obj):
# 	"""return EXPANDABLE version of obj?
# 	no, can't bake in 'only one level'
# 	of immutability
# 	or maybe we can, maybe that's the most sane thing to do
# 	"""
# 	if isinstance(obj, PlugBase):
# 		obj = obj.MPlug
# 	if isinstance(obj, om.MPlug):
# 		return _getElementsMPlug(obj)
# 	# if isinstance(obj, (tuple, list)):
# 	# 	return obj
# 	# return (obj, )
# 	if isinstance(obj, (tuple, list)):
# 		return list(obj)
# 	return [obj]
# """
# for arrays -
# check if plug has sparse indices - eg if max logical index > max physical index
# if yes, treat it as a map -
# if no, treat it as a dense sequence
#
# """
#
# def _isImmutable(obj):
# 	return isinstance(obj, (str, tuple, float, int, om.MMatrix, om.MPlug))
#
# def _complexSourceMatchesLeafTarget(possible, leaf):
# 	"""MAYBE???
# 	for the case of floatArray plugs, where a normally non-leaf value
# 	might match a specific leaf target"""
# 	return False
#
# def _complexSidesMatchDirect(src, dst):
# 	""" check if 2 arbitrary complex objects
# 	can easily be said to match -
# 	EG if you have 2 compound attributes with the same structure.
#
# 	in this case, this pair is yielded and no more recursion done
# 	"""
# 	return False
#

