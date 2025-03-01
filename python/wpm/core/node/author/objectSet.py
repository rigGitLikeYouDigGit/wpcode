from __future__ import annotations
import typing as T

from ..gen.objectSet import ObjectSet as GenObjectSet

from wpm import cmds, om, WN

class ObjectSet(GenObjectSet):
	""" wrapper for adding things to node sets in a sane way """

	clsApiType = om.MFn.kSet

	def add(self, target):
		"""add given node to this set"""
		cmds.sets(target, edit=1, add=self)

	def addChildNodes(self, newChildren, relative=True):
		"""set hierarchy, not dag hierarchy"""
		cmds.sets(newChildren, edit=1, include=self)

	def objects(self)->T.Set[WN]:
		items = cmds.sets( self, q=True )
		if not items: return set()
		return set([WN(i) for i in items])

	def sets(self)->T.Set[ObjectSet]:
		return set(filter(lambda i : isinstance(i, ObjectSet), self.objects()))


