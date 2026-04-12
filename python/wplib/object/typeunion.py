from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wplib.object.wraplist import WrapList


class UnionWrapList(WrapList):
	"""add method to getAvailableMembers or something, to control
	what shows up as the union here"""
	pass

class IntersectionWrapList(WrapList):
	pass

class Unionable(type):
	"""metaclass mixin to allow unioning/intersecting
	types with bool expressions, and retrieving
	values from class methods that correspond to the result.

	so classmethod on a union type would return union of all results,
	on intersect type would return intersection, etc


	There's exactly one use case for this I can think of, but it
	was a fun idea
	"""

	def __or__(cls, other):
		if isinstance(other, Unionable):
			return WrapList()


