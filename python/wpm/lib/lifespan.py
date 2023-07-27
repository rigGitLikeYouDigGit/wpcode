
from __future__ import annotations
import typing as T

from wpm import om

"""simple class tied to lifespan of specific maya node - 
can be a mixin, 
could probably be a subclass of some abstract Lifespan object
"""


class NodeLifespanTracker:
	"""simple class tied to lifespan of specific maya node -
	can be a mixin,
	could probably be a subclass of some abstract Lifespan object
	"""

	def __init__(self, node: om.MObject):
		self._node : om.MObject = None
		self.attachToNode(node)

	def attachToNode(self, node: om.MObject):
		"""attach to node, set up callbacks"""
		self._node = node
		self.onAttach()

	def node(self) -> om.MObject:
		"""return node"""
		return self._node

	def onAttach(self):
		"""called when tracker is attached to node"""
		pass

	def onNodeObjectCreated(self):
		"""called when node object is created, before any other callbacks -
		only triggers once per node object, deleting a node and then undoing still
		preserves the MObject, so this will not trigger again"""
		pass

	def onNodeAddedToScene(self):
		"""called when node is added to scene, triggered by creating and redo"""
		pass

	def onNodeRemovedFromScene(self):
		"""called when node is removed from scene, triggered by deleting and undo"""
		pass

	def onNodeObjectDestroyed(self):
		"""called when node object is fully deleted, undo queue flushed"""
		pass




