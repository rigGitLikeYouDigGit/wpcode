
from __future__ import annotations
import typing as T

from wpm import om
from wpm.core import CallbackOwner

"""simple class tied to lifespan of specific maya node - 
can be a mixin, 
could probably be a subclass of some abstract Lifespan object
"""


class NodeLifespanTracker(CallbackOwner):
	"""simple class tied to lifespan of specific maya node -
	can be a mixin,
	could probably be a subclass of some abstract Lifespan object.

	_paused status is set when node is removed from graph, but still
	exists as an MObject in undo history - check this status in subclasses for
	more complex callback behaviours

	"""

	LIFESPAN_KEY = "lifespan"

	def __init__(self, node: om.MObject):
		CallbackOwner.__init__(self)
		self._nodeHandle : om.MObjectHandle = None
		self._paused = False
		self.attachToNode(node)


	def node(self) -> om.MObject:
		"""return node"""
		return self._nodeHandle.object()

	def isPaused(self) -> bool:
		"""return paused status"""
		return self._paused


	def attachToNode(self, node: om.MObject):
		"""attach to node, set up callbacks"""
		self.removeAllOwnedCallbacks()
		self._nodeHandle = om.MObjectHandle(node)

		self._linkLifespanCallbacks(node)
		# self.onAttach()

	def _onNodeAddedToSceneCheck(self, *args, **kwargs):
		"""check if node added is this node -
		skip if not.
		We could probably use a singleton to listen to dg events and filter them,
		but for plugins I prefer it being contained"""
		if args[0] == self.node():
			self.onNodeAddedToScene(*args, **kwargs)

	def _linkLifespanCallbacks(self, node: om.MObject):

		# link to destruction function
		self.addOwnedCallback(
			om.MNodeMessage.addNodeDestroyedCallback(
				node, self.afterNodeObjectDestroyed),
			key=self.LIFESPAN_KEY
		)

		# node creation
		# MModelMessage freezes maya when you try to add a callback with it
		# so we need to get
		# c r e a t i v e

		self.addOwnedCallback(
			om.MNodeMessage.addNodePreRemovalCallback(
				node, self.onNodeRemovedFromScene),
			key=self.LIFESPAN_KEY
		)

		self.addOwnedCallback(
			om.MDGMessage.addNodeAddedCallback(
				self._onNodeAddedToSceneCheck,
				om.MFnDependencyNode(node).typeName),
			key=self.LIFESPAN_KEY
			)


	def _detachLifespanCallbacks(self):
		self.removeAllCallbacksWithKey(self.LIFESPAN_KEY)

	def pause(self):
		"""pause callbacks"""
		#print("pausing tracker")
		self._paused = True

	def resume(self):
		"""resume callbacks"""
		#print("unpausing tracker")
		self._paused = False


	def onAttach(self):
		"""OVERRIDE :
		called when tracker is attached to node"""
		pass

	# def onNodeObjectCreated(self):
	# 	"""called when node object is created, before any other callbacks -
	# 	only triggers once per node object, deleting a node and then undoing still
	# 	preserves the MObject, so this will not trigger again"""
	# 	pass
	# and obviously you can't actually attach to a node that doesn't exist yet :)

	def onNodeAddedToScene(self, *args, **kwargs):
		"""called when node is added to scene, triggered by creating and redo"""
		#print("node added to scene")
		self.resume()
		pass

	def onNodeRemovedFromScene(self, *args, **kwargs):
		"""called when node is removed from scene, triggered by deleting and undo
		by default, pause processing if node is deleted"""
		#print("node removed from scene")
		self.pause()
		pass

	def afterNodeObjectDestroyed(self, *args, **kwargs):
		"""OVERRIDE :
		called when node object is fully deleted, undo queue flushed.

		I think we can just set this to remove all callbacks
		"""
		#print("node object destroyed")
		self.removeAllOwnedCallbacks()


