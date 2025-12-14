from __future__ import annotations
import pprint
import traceback, time

from importlib import reload

import hou
from hou import qt, undos
from PySide6 import QtWidgets, QtCore, QtGui

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
	print("on node created")
	pprint.pprint(args)
	pprint.pprint(kwargs)
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

def onNodeInternalChanged(node:hou.Node, *args, **kwargs):
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

def onNodeOperationErrored(*args, **kwargs):
	"""automatically press undo after node errors -
	this might be a bad idea but still better than an invalid state"""
	timer = QtCore.QTimer()
	timer.singleShot(100, lambda *a, **k : undos.performUndo())


"""
functions managing locally defined HDAs in file
"""
def onNewLeafPressed(node:hou.Node, *args, **kwargs):
	"""when user presses New Leaf, create new
	locally saved hda for just this node, with whatever name
	has been given for it"""


def onNodeDefSelected(node:hou.Node, *args, **kwargs):
	""" - set button to "Edit leaf",

	if no "final" hda found for combination of parent bases, create it

	then populate parent params and add "override" checkbox under each one

	"""
	pass

def getHDAsDefinedInScene()->list[hou.HDADefinition]:
	"""return list of all scene-bound hdas.
	baked textHDA hdas are always going to be scene-bound"""

def pullLocalNodeState(node:hou.Node):
	hda = TextHDANode(node)

	with hda.workCtx() as ctx:
		# get stored incoming node state
		storedIncomingState = hda.getCachedParentState()
		#print("stored incoming state:")
		#pprint.pprint(storedIncomingState)

		# get whole state of node in scene
		wholeNodeState = gather.getFullNodeState(node)
		#print("wholeNodeState:")
		#pprint.pprint(wholeNodeState, depth=5)
		# return

		# get current delta
		leafDelta = gather.diffNodeState(storedIncomingState, wholeNodeState)
		#print("leaf delta:")
		#pprint.pprint(leafDelta)
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
	if leafDelta not given, pull from node params

	if allowEditing is False, don't include leaf delta in
	final state
	"""

	hda = TextHDANode(node)

	if leafDelta is None:
		leafDelta = hda.nodeLeafStoredState()
		if hda.parentPaths() and not hda.editingAllowed():
			leafDelta = {}

	errored = False
	# with hou.undos.group("syncTextHDA") as undoGroup:
	try:

		# return
		# reload parent states from files on node
		newParentState = hda.reloadParentStates()
		#print("new parent state:")
		#pprint.pprint(newParentState)

		# reapply leaf delta on to it
		if leafDelta:
			fullNodeState = mergeNodeStates(newParentState, [leafDelta])
		else:
			fullNodeState = newParentState
		#print("fullNodeState:")
		#pprint.pprint(fullNodeState)

		# now sync node state to whole node state
		gather.setNodeToState(node, fullNodeState)
	except Exception as e:
		traceback.print_exc()
		errored = True
		print("ERRORED")
		onNodeOperationErrored(node)
	return errored

def onResetToBasesBtnPressed(node:hou.Node):
	pass

def onHardResetBtnPressed(node:hou.Node):
	hdaNode = TextHDANode(node)
	hdaNode.fullReset()
	node.parm(ParmNames.allowEditing).set(False)
	pass

def onParentDefNameChanged(node:hou.Node, parm:hou.Parm):
	pass

def onParentDefVersionChanged(node:hou.Node, parm:hou.Parm):
	pass

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
	with hda.workCtx():
		leafDelta = pullLocalNodeState(node)
		refreshParentBasesRegenNode(node, leafDelta)


def onClearLeafPressed(node, *args, **kwargs):
	"""remove local data from node"""
	hda = TextHDANode(node)
	with hda.workCtx():
		hda.nodeLeafDeltaParm().set("")
		refreshParentBasesRegenNode(node)


def onAllowEditingChanged(node:hou.Node, *args, **kwargs):
	"""create a new embedded hda def
	and open up node for editing

	"""
	hda = TextHDANode(node)
	hda.getHDADef() # create new def if it doesn't exist
	if hda.editingAllowed():
		node.allowEditingOfContents(False)
		return
	else:
		# just freeze local contents
		node.matchCurrentDefinition()

	refreshParentBasesRegenNode(node)
	pass




def onLeafFilePathChanged(node:hou.Node, parm:hou.Parm, *args, **kwargs):
	"""try and extract a version from the path, set that on the version menu if not the latest one
	if allowEditing is not checked, conform node to new path
	"""
	hda = TextHDANode(node)
	newPath = hda.nodeLeafPath()
	if hda.editingAllowed(): # local deltas mutable
		if not newPath.is_file():
			return
		return
	# editing not allowed, remove local deltas to what is defined in file, conform state
	if not newPath.is_file():
		node.addWarning(f"No file found at path {newPath}, cannot import new leaf state")
		return
	try:
		loads(newPath.read_text())
	except:
		node.addError(f"Invalid json at path {newPath}, cannot import new state")
	hda.nodeLeafDeltaParm().set(newPath.read_text())
	try:
		refreshParentBasesRegenNode(node)
	except Exception as e:
		node.addError("ERROR regenerating node: \n" + traceback.format_exc())
		pass

def onLeafTextChanged(node:hou.Node, parm:hou.Parm, *args, **kwargs):
	"""don't necessarily do anything"""
	pass


def onParentFilePathChanged(node:hou.Node, parm:hou.Parm, *args, **kwargs):
	"""if no leaf data is set, disable editing by default -
	this is a newly created node being directly set to existing definition"""
	hda = TextHDANode(node)
	hda.reloadParentStates()
	if not hda.nodeLeafStoredState():
		node.parm(ParmNames.allowEditing).set(False)
	pass

def onParentTextChanged(node:hou.Node, parm:hou.Parm):
	"""editing a parent's text by hand isn't allowed, this function
	won't do anything"""
	pass


def onSaveBtnPressed(node:hou.Node):
	"""write current node delta to target file -
	"""
	hda = TextHDANode(node)
	if not hda.nodeLeafPath():
		node.addError("No leaf path to save local deltas")
		return
	hda.nodeLeafPath().write_text(
		hda.nodeLeafDeltaParm().eval()
	)


def onLeafDownBtnPressed(node:hou.Node):
	"""push leaf on to end of parent bases"""


def onLeafUpBtnPressed(node: hou.Node):
	"""pull first parent base up to leaf for editing"""






