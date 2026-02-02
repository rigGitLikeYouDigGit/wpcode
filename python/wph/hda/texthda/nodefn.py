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
from .gather import mergeNodeStates, TextHDANode, dbg
from .types import ParmNames, CachedFile, loads, dumps
"""direct functions called by textHDA node
TODO: we duplicate a bit between onDefChanged and onAllowEditingChanged - 
have some overall sync function to take account of both of them

errors on setting editing allowed to false with node data - locking asset def,
then trying to set info inside it
unlock asset each time?
or just match to definition should be enough


setting parametres under leaf params is only allowed / preserved when editing 
allowed, otherwise they're just untracked leaf attributes as with any other
kind of hda

"""



def getMultiFolderChildParm(folderParm:hou.Parm, name:str, folderIndex=0,
                            nParmsPerFolder=3):
	folderParm.multiParmInstances()

INTERNAL_EVENT_TYPES = [
	# hou.nodeEventType.ParmTupleChanged,
	hou.nodeEventType.ChildDeleted,
	hou.nodeEventType.ChildReordered,
	hou.nodeEventType.ChildSwitched,
	#hou.nodeEventType.ChildSelectionChanged,
	hou.nodeEventType.NetworkBoxCreated,
	hou.nodeEventType.NetworkBoxChanged,
	hou.nodeEventType.NetworkBoxDeleted,
	hou.nodeEventType.StickyNoteCreated,
	hou.nodeEventType.StickyNoteChanged,
	hou.nodeEventType.StickyNoteDeleted,

	hou.nodeEventType.IndirectInputCreated,
	hou.nodeEventType.IndirectInputRewired,
	hou.nodeEventType.IndirectInputDeleted,

	hou.nodeEventType.SpareParmTemplatesChanged
]

class HoudiniCallbackFn:
	"""using callable objects for easier identification, since
	we can't otherwise distinguish between 2 callbacks set for the
	same event type.

	this means you need a 2-step lookup to get the top node and
	kwargs from this object in the actual function, but we live with it
	"""
	def __init__(self, fn, *args, **kwargs):
		self.fn = fn
		self.args = args
		self.kwargs = kwargs
	def __call__(self, *args, **kwargs):
		self.fn(*args, callbackObj=self, **kwargs)

def isinstanceReloadSafe(obj, types):
	""" got that junior drip
	got that day-4 python dev drip
	quirked up intern boy with a little swag
	"""
	if not isinstance(types, (tuple, list)):
		types = (types,)
	return any(type(obj).__name__ in (base.__name__ for base in i.__mro__)
	           for i in types
	)

NODE_INTERNAL_CB_NAME = "nodeInternalCallback"

@dbg
def addNodeInternalCallbacks(node:hou.OpNode, inputRewired=False,
                             paramChanged=True,
                             topNode:hou.OpNode=None):
	print("addNodeInternalCallbacks:", node)
	typesToAdd = INTERNAL_EVENT_TYPES
	cb = HoudiniCallbackFn(
		onNodeInternalChanged, topNode=topNode,
		name=NODE_INTERNAL_CB_NAME
	)
	if inputRewired:
		typesToAdd = typesToAdd + [hou.nodeEventType.InputRewired]
	if paramChanged: # don't set this on top node
		typesToAdd = typesToAdd + [hou.nodeEventType.ParmTupleChanged]

	node.addEventCallback(
		tuple(typesToAdd), cb
	)
	# for i in typesToAdd:
	# 	node.addEventCallback(
	# 		(i,),
	# 		lambda *args, **kwargs : onNodeInternalChanged(
	# 			*args,
	# 		           topNode=topNode or node,
	#                    **kwargs)
	# 	)

@dbg
def removeNodeInternalCallbacks(node:hou.OpNode):
	#print("removeNodeInternalCallbacks:", node)
	for eventTypes, cbFn in tuple(node.eventCallbacks()):
		# print("check", eventTypes, cbFn, isinstanceReloadSafe(cbFn,
		#                                                 HoudiniCallbackFn))
		if isinstanceReloadSafe(cbFn, HoudiniCallbackFn) and cbFn.kwargs["name"] == NODE_INTERNAL_CB_NAME:
			#print("removing callback:", eventTypes, cbFn)
			node.removeEventCallback(eventTypes, cbFn)

	# print("events after:")
	# for i in node.eventCallbacks():
	# 	print(i)


PARENT_LEAF_PARAM_CB_NAME = "parentLeafCallback"

@dbg
def addHDAParamCallbacks(node:hou.OpNode):
	"""add callback to trigger whenever leaf or overridden params change
	on this node
	"""
	callback = HoudiniCallbackFn(
		onHDAParentOrLeafParamChanged,
		name=PARENT_LEAF_PARAM_CB_NAME
	)
	cbParmNames = []
	hda = TextHDANode(node)
	for leafPt in hda.leafHDAParmTemplates():
		cbParmNames.append(leafPt.name())
	# for parentPt in hda.parentHDAParmTemplates():
	# 	if "PARENT_" in parentPt.name() or "OVERRIDE_" in parentPt.name():
	# 		continue

	callback.kwargs["cbParmNames"] = cbParmNames
	node.addParmCallback(
		callback,
		tuple(cbParmNames)
	)

@dbg
def removeHDAParamCallbacks(node:hou.OpNode):
	for i in node.eventCallbacks():
		if isinstanceReloadSafe(i[1], HoudiniCallbackFn):
			if i[1].kwargs["name"] == PARENT_LEAF_PARAM_CB_NAME:
				node.removeEventCallback(i[0], i[1])

@dbg
def syncHDAParamCallback(node:hou.OpNode):
	"""add callback to trigger whenever leaf or overridden params change
	on this node

	parent OVERRIDDEN param only acts at node level -
		if we want further nodes to inherit those overrides, then
		node has to be made a new def, and those parms moved up to leaf section

	so we actually don't care at all about parent params?
	"""
	removeHDAParamCallbacks(node)
	addHDAParamCallbacks(node)

@dbg
def onHDAParentOrLeafParamChanged(node,
                                  callbackObj:HoudiniCallbackFn=None,
                                  **kwargs):
	"""runs whenever leaf or parent param in folder changes
	"""

@dbg
def onHDAParmTemplateChanged(node:hou.OpNode, *args, callbackObj=None, **kwargs):
	"""runs whenever a leaf or parent parm template changes -
	discard IMMEDIATELY unless it concerns leaf"""



@dbg
def onNodeCreated(node:hou.OpNode, *args, **kwargs):
	"""attach callback to node, add it to the
	main TextHDA node bundle"""
	print("onNodeCreated:", node)

	bundle = gather.getTextHDANodeBundle()
	bundle.addNode(node)
	hdaNode = TextHDANode(node)
	node.setExpressionLanguage(hou.exprLanguage.Python)
	hdaNode.setWorking(False)
	with hdaNode.workCtx():
		if not hdaNode.defFileParm().evalAsString():
			hdaNode.editingAllowedParm().disable(True)
		print("textHda created:", node)
		print("nodes in bundle:", gather.allSceneTextHDANodes())

@dbg
def onNodeLoaded(node:hou.OpNode, *args, **kwargs):
	"""runs when node loaded during opening scene - apparently
	either this or onNodeCreated runs, never both"""
	print("onNodeLoaded:", node)

	bundle = gather.getTextHDANodeBundle()
	bundle.addNode(node)

	hdaNode = TextHDANode(node)
	hdaNode.setWorking(False)
	node.setExpressionLanguage(hou.exprLanguage.Python)
	with hdaNode.workCtx():
		if not hdaNode.defFileParm().evalAsString():
			hdaNode.editingAllowedParm().disable(True)

	#onAllowEditingChanged(node)

"""analysing hda section infos:
nothing in either ExtraFileOptions or InternalFileOptions

"""
@dbg
def onNodeDeleted(node:hou.Node, type:hou.OpNodeType, *args, **kwargs):
	hdaDef : hou.HDADefinition = type.definition()
	print("on node deleted:", node, hdaDef)
	if hdaDef == gather.getBaseTextHDADef():
		print("is base def, skipping")
		return
	# get instances
	if not type.instances():
		print("not instances, destroying")
		hdaDef.destroy()

@dbg
def onNodeLastDeleted(*args, **kwargs):
	"""called after the last instance of an HDA definition is deleted from
	scene; NB this includes DERIVED definitions too.
	So here we check if node currently uses a definition other than the master
	textHDA, and if so, delete that definition from the scene
	"""
	nodeType : hou.NodeType = kwargs["type"]
	hdaDef : hou.HDADefinition = nodeType.definition()
	print("on node last deleted:", hdaDef)
	if hdaDef == gather.getBaseTextHDADef():
		print("is base def, skipping")
		return
	print("deleting")
	hdaDef.destroy()

@dbg
def onNodeNameChanged(node:hou.Node, *args, **kwargs):
	"""here's a silly(?) idea - since we give every textHDA node its own HDA
	definition, link the name of the definition
	to the node itself.
	otherwise if no local edits are present, reset to base def
	"""
	hda = TextHDANode(node)
	if hda.filteredParentStoredStates() or hda.editingAllowed():
		gather.createLocalTextHDADefinition(node, node.name() + "_TextHDA")
	else:
		hda.fullReset(resetParms=False)

@dbg
def onHDAUpdated(kwargs):
	"""callback whenever HDA definition is updated
	THIS should be the only time that HDA
	parmtemplates are ever updated in leaf data -
	I think this means we have to update ALL node
	instances in the scene?
	"""
	hdaType : hou.SopNodeType = kwargs["type"]
	instances = hdaType.instances()
	defStr = ""
	for i in instances:
		hdaNode = TextHDANode(i)
		defStr = hdaNode.defFile()
		# leafParmTemplates = hdaNode.leafHDAParmTemplates()
		# leafParmTemplateStr = {
		# 	i.name() : i. for i in leafParmTemplates
		# }
		leafParmTemplateStr = hdaNode.leafPTGData()
		leafState = hdaNode.nodeLeafStoredState()
		if not leafState:
			continue
		print("leaf state")
		pprint.pprint(leafState)
		leafState["parmTemplates"].setdefault("LEAF", {})["LEAF"] = (
			leafParmTemplateStr)
		hdaNode.nodeLeafDeltaParm().set(dumps(leafState))
	"""check through dependent defs to update - we only have to live update 
	nodes in the actual scene, since building a node from a file will
	automatically get updates"""
	dependentNodes = gather.getDefAffectedNodeMap()[defStr]
	for i in dependentNodes:
		if i.type() == hdaType:
			continue
		hdaNode = TextHDANode(i)
		hdaNode.syncNodeState()




def onChildNodeCreated(rootTextNode:hou.Node, *args, **kwargs):
	"""also propagate callback to children contained"""

@dbg
def onNodeInternalChanged(
		node:hou.Node,
		event_type:hou.nodeEventType,
		#topNode:hou.OpNode,
		*args, **kwargs):
	"""
	this should ONLY fire if node is open for editing

	callback whenever something internal changes on node
	check if node live checkbox is ready

	split into 2 stages, pulling and diffing local node, and updating node from parent states -
	auto update only does local pull

	node is whatever callback is called on; topNode is the top TextHDA to use for deltas

	"""
	cbObj : HoudiniCallbackFn = kwargs["callbackObj"]
	topNode = cbObj.kwargs["topNode"]
	#print("node internal changed:", topNode, node, event_type, args, kwargs)
	hda = TextHDANode(topNode)
	if hda.isWorking():
		print("working, skipping internal", node)
		return

	# if allowEditing, skip, it's already taken care of in separate callback
	if event_type == hou.nodeEventType.ParmTupleChanged:
		# sometimes it can be a parmtuple event with parmtuple given as None
		if kwargs.get("parm_tuple"):
			parm_tuple : hou.ParmTuple = kwargs["parm_tuple"]
			if parm_tuple.name() == ParmNames.allowEditing:
				return
		if node == topNode:
			return

	if not hou.node(node.path()): # already deleted
		return
	if not hou.node(topNode.path()): # already deleted
		return
	with hda.workCtx():
		# if new node created
		if event_type == hou.nodeEventType.ChildCreated:
			childNode : hou.OpNode = kwargs["child_node"]
			print("child node created:", childNode)
			addNodeInternalCallbacks(childNode, inputRewired=True, topNode=topNode)
		pullLocalNodeStateAndUpdateDef(topNode)

		# update other nodes in scene
		dependencyMap = gather.getDefAffectedNodeMap()
		print("affected node map:")
		pprint.pprint(dependencyMap)
		for i in dependencyMap[hda.defFile()]:
			if i.path() == node.path():
				continue
			otherTextHda = TextHDANode(i)
			print("SYNC DEPENDENT NODE:", i)
			otherTextHda.syncNodeState()


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

@dbg
def pullLocalNodeState(node:hou.Node)->tuple[dict, dict, dict]:
	"""return
	(parent incoming state, whole node state, leaf delta)
	"""
	hda = TextHDANode(node)
	# get stored incoming node state
	storedIncomingState = hda.reloadParentStates()

	# get whole state of node in scene
	wholeNodeState = gather.getFullNodeState(node)
	# get current delta
	leafDelta = gather.diffNodeState(storedIncomingState, wholeNodeState)
	return storedIncomingState, wholeNodeState, leafDelta

# print("leaf delta:")
# pprint.pprint(leafDelta, depth=5)
# # save on node

@dbg
def pullLocalNodeStateAndUpdateDef(node:hou.Node):
	hda = TextHDANode(node)
	if not hda.hdaDef():
		return {}

	storedIncomingState, wholeNodeState, leafDelta = pullLocalNodeState(node)

	"""move leaf parm templates to parent"""

	mergedState = gather.mergeNodeStates(storedIncomingState)
	gather.setNodeToState(node, )

	with hda.workCtx() as ctx:
		hda.setNodeLeafDeltaData(leafDelta)
		gather.updateHDADefSections(hda.hdaDef(),
		                            wholeNodeState,
		                            leafDelta,)

		#print("saved leaf deltas on node")
		#print(toSave)
		return leafDelta

@dbg
def pushLocalNodeState(node:hou.OpNode):
	"""why are you writing functions at 4 in the morning
	because i have lost control of my life

	pull local state, then push out to all dependents active in scene
	"""
	hda = TextHDANode(node)
	#pullLocalNodeStateAndUpdateDef(node)
	print("affected:")
	pprint.pprint(gather.getDefAffectedNodeMap())
	for i in gather.getDefAffectedNodeMap()[hda.defFile()]:
		print("push to", i, i.type(), i.type().definition() == node.type().definition())
		if i.type().definition() == node.type().definition():
			continue
		hda = TextHDANode(i)
		hda.syncNodeState()

@dbg
def pushNodeParamValues(node:hou.Node):
	"""like above but only update param values"""
	hda = TextHDANode(node)




@dbg
def refreshParentBasesRegenNode(node:hou.Node, leafDelta:dict=None)->bool:
	"""sync parent bases, regenerate node data
	if leafDelta not given, pull from node params

	if allowEditing is False, don't include leaf delta in
	final state
	"""

	hda = TextHDANode(node)

	if leafDelta is None:
		leafDelta = hda.nodeLeafStoredState()
		if hda.parentDefPaths() and not hda.editingAllowed():
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

@dbg
def onHardResetBtnPressed(node:hou.Node):
	hdaNode = TextHDANode(node)
	with hdaNode.workCtx():
		hdaNode.fullReset()
		node.parm(ParmNames.allowEditing).set(False)
	pass

def onLiveUpdateChanged(node:hou.OpNode, *args, **kwargs):
	pass

@dbg
def onParentDefNameChanged(node:hou.Node, kwargs):
	print("parent def name changed:", node, kwargs)
	hda = TextHDANode(node)
	hda.reloadParentStates()
	gather.setNodeToState(node, hda.getCachedParentState())
	pass

def onParentDefVersionChanged(node:hou.Node, parm:hou.Parm):
	pass

@dbg
def onSyncBtnPressed(node:hou.OpNode):
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
	hda = TextHDANode(node)
	pullLocalNodeStateAndUpdateDef(node)
	pushLocalNodeState(node)


def onClearLeafPressed(node, *args, **kwargs):
	"""remove local data from node"""
	hda = TextHDANode(node)
	with hda.workCtx():
		hda.nodeLeafDeltaParm().set("")
		refreshParentBasesRegenNode(node)

@dbg
def onDefFileLineChanged(node:hou.Node, kwargs):
	"""check if file is valid, try and resolve valid one etc
	set new value on def file line if possible, if not,
	set warning
	ok FINE we test now what happens if you don't pass a full file path
	here - emulates a scene-bound definition,
	changing the name of the HDA created.

	if editing not allowed, add this node to bundle of nodes affected
	by the path def
	if editing IS allowed, add node to bundles of all parent defs
	"""
	print("on defFileLineChanged")
	hdaNode = TextHDANode(node)
	if hdaNode.isWorking():
		print("node", node, "still working")
		return
	with hdaNode.workCtx() as ctx:
		newDefStr = hdaNode.defFileParm().evalAsString().strip()
		if newDefStr:
			hdaNode.editingAllowedParm().disable(False)
			# rename current hda def if needed
			defName = gather.hdaDefNameFromDefFile(
				newDefStr, edit=hdaNode.editingAllowed())
			hdaDef, node = gather.createLocalTextHDADefinition(
				node, defName)
			hdaNode = TextHDANode(node)
			ctx.path = node.path()

			if hdaNode.editingAllowed():
				pass
			else: # match definition and prevent editing
				node.matchCurrentDefinition()

		else: # prevent local edits if no def file set

			hdaNode.fullReset(resetParms=False)
			hdaNode.editingAllowedParm().set(False)
			hdaNode.editingAllowedParm().disable(True)
	pass

def onSelectDefBtnPressed(node:hou.Node, kwargs):
	print("selectDef btn pressed")

@dbg
def onAllowEditingChanged(node:hou.OpNode, *args, **kwargs):
	"""create a new embedded hda def
	and open up node for editing

	need to flag that this signal can't fire while it's already executing

	we do still need a custom hda def for each node, even just for displaying
	parent bases

	"""
	hda = TextHDANode(node)

	if hda.isWorking():
		print("node still working, aborting")
		return

	try:
		node.path()
	except hou.ObjectWasDeleted:
		return
	with hda.workCtx():
		node = hda.node
		if hda.editingAllowed():
			print("editing allowed")
			# required so we register nodes being created in hda
			addNodeInternalCallbacks(node, inputRewired=False,
			                         paramChanged=False,
			                         topNode=node)
			syncHDAParamCallback(hda.node)
			node.allowEditingOfContents(False)
			for i in node.allSubChildren(recurse_in_locked_nodes=False):
				addNodeInternalCallbacks(i, inputRewired=True, topNode=node)

			return
		else:
			print("edit not allowed")
			removeNodeInternalCallbacks(node)
			for i in node.allSubChildren(recurse_in_locked_nodes=False):
				removeNodeInternalCallbacks(i)
			# just freeze local contents
			if hda.hasIncomingStates():
				print("incoming states found")
				hdaDef = hda.getCustomHDADef()
				hdaDef.updateFromNode(node)
				node.matchCurrentDefinition()
				#refreshParentBasesRegenNode(node)

			else: # nothing in any text params
				if hda.defFile():
					node.matchCurrentDefinition()
					#hda.fullReset(resetParms=False)
					pass
				else: # no def file given, fully reset
					hda.fullReset(resetParms=True)

def getDefMenuItems(kwargs)->list[str]:
	"""
	look over all definitions under files and in scene
	need to return list of
	["value1", "label1", "value2", "label2"]
	"""
	#print("getDefMenuItems", kwargs)
	hda = TextHDANode(kwargs['node'])
	parm : hou.Parm = kwargs['parm']
	result = []
	# don't list node's own def as available
	currentDef = hda.defFile()

	sceneDefs = gather.getSceneHDADefNodes()
	#print("sceneDefs", sceneDefs)
	if currentDef in sceneDefs:
		sceneDefs.pop(currentDef)
	if sceneDefs:
		result.extend(["", "==SCENE DEFS=="])
		for defName, nodes in sceneDefs.items():
			result.extend([defName, defName])

	fileDefs = gather.getFileHDADefs()
	#print("fileDefs", fileDefs)
	if currentDef in fileDefs:
		sceneDefs.pop(currentDef)
	if fileDefs:
		result.extend(["", "==FILE DEFS=="])
		for defName, versionMap in fileDefs.items():
			latest = list(versionMap.values())[-1]
			result.extend([defName, latest])

	return result



@dbg
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

@dbg
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


def onClearUserDataBtnPressed(node:hou.Node, *args, **kwargs):
	"""debug, clear all user data from the given node, in case
	a node gets stuck thinking it's still working
	"""
	node.destroyUserData(
		"_working", must_exist=False
	)

def onLeafDownBtnPressed(node:hou.Node):
	"""push leaf on to end of parent bases"""


def onLeafUpBtnPressed(node: hou.Node):
	"""pull first parent base up to leaf for editing"""






