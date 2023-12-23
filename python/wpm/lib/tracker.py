
from __future__ import annotations
import typing as T

from wpm import om
from wpm.core import CallbackOwner, getMObject
from wpm.lib import hierarchy as h

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

	def dagPath(self) -> om.MDagPath:
		"""return dag path to node"""
		return om.MDagPath.getAPathTo(self.node())

	def isPaused(self) -> bool:
		"""return paused status"""
		return self._paused


	def attachToNode(self, node: om.MObject):
		"""attach to node, set up callbacks"""
		node = getMObject(node)
		self.removeAllOwnedCallbacks()
		self._nodeHandle = om.MObjectHandle(node)

		self._linkLifespanCallbacks(node)
		self.onAttach()

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
		self.pause()
		pass

	def afterNodeObjectDestroyed(self, *args, **kwargs):
		"""OVERRIDE :
		called when node object is fully deleted, undo queue flushed.

		I think we can just set this to remove all callbacks
		"""
		#print("node object destroyed")
		self.removeAllOwnedCallbacks()



class NodeHierarchyTracker(NodeLifespanTracker):
	"""track hierarchy of nodes
	generate events whenever any child is modified, added or removed
	can probably get by without multiple tracker objects,
	just filter commands with this one

	found some VERY deep lore from maya,
	__PrenotatoPerDuplicare
	__PrenotatoPerDuplicare!
	best guess is at some point during a duplication operation, a newly created
	node gets a prefix to make sure it's unique while a proper name is found (since
	it probably needs to be incremented from existing siblings)

	There's also some weird events of kScaleX and kScaleY before and after a node is
	reparented, but I'm not sure what those are for.

	This node defers the NodeRemoved function til the change has been made to the dag
	graph - this is done by holding two flags on this object as state. Not great at all,
	but I think it's ok.

	We don't expose any complex filtering or whitelisting for now - when this tracks
	nodes in a hierarchy it tracks EVERYTHING

	TODO: add a filter for node types, and a whitelist for node names
	TODO: add mechanism to limit frequency of specific callbacks (like mesh editing)

	TODO: keep a tree updated with node hierarchy, and provide a way to query it,
	with branch values being DATA OBJECTS for that node

	"""

	CHILD_CALLBACK_KEY = "childCallback"

	def __init__(self, node:om.MObject):
		super(NodeHierarchyTracker, self).__init__(node)
		#self._childTrackers : list[NodeLifespanTracker] = []
		self._shouldSyncOnAdded = False
		self._oldRemoveChildParentPath : tuple[om.MDagPath, om.MDagPath] = ()

	# def onAnyDagChange(self, *args, **kwargs):
	# 	code = args[0]
	# 	codeName = classConstantValueToNameMap(om.MDagMessage)[code]
	# 	print("any dag change", codeName,
	# 	      om.MFnDagNode(args[1]).name(), ",",
	# 	      om.MFnDagNode(args[2]).name(),
	# 	      )

	def onAttach(self):
		"""set up filter callbacks"""
		super(NodeHierarchyTracker, self).onAttach()
		# self.addOwnedCallback(
		# 	om.MDagMessage.addAllDagChangesCallback(self.onAnyDagChange)
		# )

		self.addOwnedCallback(
			om.MDagMessage.addChildAddedCallback(self._onAnyChildAdded)
		)
		self.addOwnedCallback(
			om.MDagMessage.addChildRemovedCallback(self._onAnyChildRemoved)
		)
		self.addOwnedCallback(
			om.MDagMessage.addChildReorderedCallback(self._onAnyChildReordered)
		)

		# set up child callbacks
		self.syncChildNodeCallbacks()


	def syncChildNodeCallbacks(self):
		"""add specific contracts for all child nodes"""
		self.removeAllCallbacksWithKey(self.CHILD_CALLBACK_KEY)
		for child in h.iterDagChildren(self.node(), includeRoot=True):

			# should probably delegate to a separate system to set up
			# specific events for node types - having everything here
			# is not great

			self.addOwnedCallback(
				om.MNodeMessage.addNameChangedCallback(
					child, self.onChildNameChanged
				),
				key=self.CHILD_CALLBACK_KEY
			)

			if child.hasFn(om.MFn.kTransform):
				self.addOwnedCallback(
					om.MDagMessage.addMatrixModifiedCallback(
						om.MDagPath.getAPathTo(child), self.onChildLocalMatrixModified
					),
					key=self.CHILD_CALLBACK_KEY
				)
			elif child.hasFn(om.MFn.kMesh):
				self.addOwnedCallback(
					om.MNodeMessage.addNodeDirtyPlugCallback(
						child, self.onChildMeshModified
					)
				)
			elif child.hasFn(om.MFn.kNurbsCurve):
				self.addOwnedCallback(
					om.MNodeMessage.addNodeDirtyPlugCallback(
						child, self.onChildCurveModified
					)
				)


	def onChildNameChanged(self, node:om.MObject, prevName:str, *args):
		"""OVERRIDE : called when a child node's name is changed"""
		#print("child name changed", node, om.MFnDependencyNode(node).name(), prevName)
		pass

	def onChildLocalMatrixModified(self, node:om.MObject, matrixModifiedFlags:int, *args):
		"""OVERRIDE : called when a child node's local matrix is modified.
		Docs say this only fires the first time the matrix is modified and dirtied -
		watch in case this gets in the way
		"""

	def createDirtyPlugFilterCallback(self, forPlug:om.MPlug,
	                               fn:T.Callable[[om.MObject, om.MPlug, ...], None],
	                               )->int:
		"""set the given function to fire only when specific plug is dirtied"""

		def _onChildPlugChangedOuter(node:om.MObject, dirtyPlug:om.MPlug, *args):
			print("child plug changed", forPlug, dirtyPlug, forPlug==dirtyPlug)
			if forPlug == dirtyPlug:
				fn(node, dirtyPlug, *args)

		return om.MNodeMessage.addNodeDirtyPlugCallback(
			forPlug.node(), _onChildPlugChangedOuter)



	def onChildMeshModified(self, node:om.MObject, dirtyPlug:om.MPlug, *args):
		"""OVERRIDE : called when a child node's poly shape is modified"""
		pass

	def onChildCurveModified(self, node: om.MObject, dirtyPlug: om.MPlug, *args):
		"""OVERRIDE : called when a child node's curve shape is modified"""
		pass

	def onChildNodeRemoved(self,
	                       newChildPath:om.MDagPath,
	                       newParentPath:om.MDagPath,
	                       oldChildPath:om.MDagPath,
	                       oldParentPath:om.MDagPath
	                       ):
		""" OVERRIDE :
		called when a child node is removed from this node -
		not called directly on the dagmessage removed callback,
		but when we detect a nodeAdded change following a nodeRemoved change.

		"""
		pass


	def onChildNodeAdded(self, childPath:om.MDagPath, parentPath:om.MDagPath):
		"""OVERRIDE : called when a child node is added to this node"""
		pass


	def onChildNodeReordered(self, childPath:om.MDagPath, parentPath:om.MDagPath):
		"""OVERRIDE : called when a child node is reordered under this node"""
		pass


	def _onAnyChildAdded(self, childDagPath:om.MDagPath, parentDagPath:om.MDagPath,
	                     *args):
		""" filter callback for child added
		check whenever a node is added to see if it falls under this node.
		Docs for MDagMessage full on lie, childdagpath is first, not second

		"""
		if self._shouldSyncOnAdded:
			self.onChildNodeRemoved(
				newChildPath=childDagPath,
				newParentPath=parentDagPath,
				oldChildPath=self._oldRemoveChildParentPath[0],
				oldParentPath=self._oldRemoveChildParentPath[1]
			)
			self._shouldSyncOnAdded = False
			self._oldRemoveChildParentPath = ()
			self.syncChildNodeCallbacks()
			# return
		if childDagPath.node() in h.iterDagChildren(self.node()):
			self.onChildNodeAdded(childDagPath, parentDagPath)
			self.syncChildNodeCallbacks()


	def _onAnyChildRemoved(self, childDagPath:om.MDagPath, parentDagPath:om.MDagPath,
	                       *args):
		"""check whenever a node is removed to see if it falls under this node
		UNFORTUNATELY this callback fires BEFORE the node is actually changed in the dag,
		so we can't do any processing on the world state -

		Here instead we set a basic flag to check on the next Added event (which is guaranteed when a child is removed)

		"""
		if childDagPath.node() in h.iterDagChildren(self.node()):
			self._shouldSyncOnAdded = True
			self._oldRemoveChildParentPath = (childDagPath, parentDagPath)


	def _onAnyChildReordered(self, childDagPath:om.MDagPath, parentDagPath:om.MDagPath,
	                         *args):
		"""check whenever a node is reordered to see if it falls under this node"""
		# check flag that may have been set by remove event
		if childDagPath.node() in h.iterDagChildren(self.node()):
			self.onChildNodeReordered(childDagPath, parentDagPath)
			self.syncChildNodeCallbacks()





