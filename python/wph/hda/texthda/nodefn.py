from __future__ import annotations
import pprint
import traceback, time

from importlib import reload

import hou

from . import gather, types
reload(gather)
reload(types)
from .gather import mergeNodeStates, TextHDANode
from .types import ParmNames, CachedFile, loads, dumps
"""direct functions called by textHDA node"""

def getMultiFolderChildParm(folderParm:hou.Parm, name:str, folderIndex=0, nParmsPerFolder=3):
	folderParm.multiParmInstances()


def onNodeCreated(node:hou.Node, *args, **kwargs):
	"""attach callback to node"""

	node.addEventCallback((
		hou.nodeEventType.ParmTupleChanged,
		hou.nodeEventType.ChildDeleted,
		hou.nodeEventType.ChildReordered,
		hou.nodeEventType.ChildSwitched,
		hou.nodeEventType.ChildSelectionChanged,
		hou.nodeEventType.NetworkBoxCreated,
		hou.nodeEventType.NetworkBoxChanged,
		hou.nodeEventType.NetworkBoxDeleted,
		hou.nodeEventType.StickyNoteCreated,
		hou.nodeEventType.StickyNoteChanged,
		hou.nodeEventType.StickyNoteDeleted,

		hou.nodeEventType.IndirectInputCreated,
		hou.nodeEventType.IndirectInputRewired,
		hou.nodeEventType.IndirectInputDeleted,

	),
		#lambda *a, **k : onNodeInternalChanged(node )
		onNodeInternalChanged
	)

	print("textHda created:", node)


def onChildNodeCreated(rootTextNode:hou.Node, *args, **kwargs):
	"""also propagate callback to children contained"""

def onNodeInternalChanged(node:hou.Node, t:int, *args, **kwargs):
	"""callback whenever something internal changes on node
	check if node live checkbox is ready

	split into 2 stages, pulling and diffing local node, and updating node from parent states -
	auto update only does local pull
	"""
	if not node.parm(ParmNames.liveUpdate).eval():
		return
	hda = TextHDANode(node)
	if hda.isWorking():
		return
	with hda.workCtx():
		pullLocalNodeState(node)



def pullLocalNodeState(node:hou.Node):
	hda = TextHDANode(node)

	with hda.workCtx() as ctx:
		# get stored incoming node state
		storedIncomingState = hda.getCachedParentState()
		print("stored incoming state:")
		pprint.pprint(storedIncomingState)

		# get whole state of node in scene
		wholeNodeState = gather.getFullNodeState(node)
		print("wholeNodeState:")
		pprint.pprint(wholeNodeState, depth=5)
		# return

		# get current delta
		leafDelta = gather.diffNodeState(storedIncomingState, wholeNodeState)
		print("leaf delta:")
		pprint.pprint(leafDelta)
		# save on node

		toSave = dumps(leafDelta)  # .replace("\n", "\\n")
		hda.nodeLeafDeltaParm().set(
			toSave
		)
		print("saved leaf deltas on node")
		print(toSave)
		return leafDelta


def refreshParentBasesRegenNode(node:hou.Node, leafDelta:dict=None)->bool:
	"""sync parent bases, regenerate node data
	if leafDelta not given, pull from node params"""

	hda = TextHDANode(node)
	if leafDelta is None:
		leafDelta = hda.nodeLeafStoredState()
	errored = False
	# with hou.undos.group("syncTextHDA") as undoGroup:
	try:

		# return
		# reload parent states from files on node
		newParentState = hda.reloadParentStates()
		print("new parent state:")
		pprint.pprint(newParentState)

		# reapply leaf delta on to it
		fullNodeState = mergeNodeStates(newParentState, [leafDelta])
		print("fullNodeState:")
		pprint.pprint(fullNodeState)

		# now sync node state to whole node state
		gather.setNodeToState(node, fullNodeState)
	except Exception as e:
		traceback.print_exc()
		errored = True
		print("ERRORED, press undo")
	return errored



def onSyncBtnPressed(node:hou.Node):
	"""sync node:
	retrieve sources, recombine deltas, reapply local deltas on node,
	then regenerate node internals


	get cached/ previous parent state
	diff scene node against THAT PREVIOUS state
	save as leaf delta
	update to latest incoming parent state
	reapply delta on top of that


	IF EDITING NOT ENABLED:
		don't touch leaf deltas, but disable them
	"""
	print("")
	print("SYNC:")
	print(":")
	hda = TextHDANode(node)

	leafDelta = pullLocalNodeState(node)

	refreshParentBasesRegenNode(node, leafDelta)




def onLeafFilePathChanged(node:hou.Node, parm:hou.Parm):
	"""try and extract a version from the path, set that on the version menu if not the latest one
	"""
	pass

def onLeafTextChanged(node:hou.Node, parm:hou.Parm):
	"""don't necessarily do anything"""
	pass


def onParentFilePathChanged(node:hou.Node, parm:hou.Parm):
	hda = TextHDANode(node)
	hda.reloadParentStates()
	pass

def onParentTextChanged(node:hou.Node, parm:hou.Parm):
	"""editing a parent's text by hand isn't allowed, this function
	won't do anything"""
	pass


def onSavePressed(node:hou.Node):
	"""write current node delta to target file -
	"""




