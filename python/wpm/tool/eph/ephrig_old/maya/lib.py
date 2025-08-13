
from edRig.palette import *
from edRig import cmds, om, EdNode
from edRig.maya.core import getMObject

from weakref import WeakValueDictionary

if T.TYPE_CHECKING:
	from edRig.ephrig.rig import EphRig, EphSolver
	from edRig.ephrig.maya.pluginnode import EphRigControlNode


nodeWrapperMap = WeakValueDictionary()


def ephPyObject(ephControlNode:T.Union[
	"EphRigControlNode", str,
	om.MObject])->"EphRigControlNode":
	"""given maya eph dependency node,
	return its associated MPxNode object
	maybe also caches it by uuid"""
	if not isinstance(ephControlNode, (str, om.MObject, EdNode)):
		return ephControlNode

	obj = getMObject(ephControlNode)
	mfnDep = om.MFnDependencyNode(obj)
	pyObj = mfnDep.userNode()
	return pyObj


def rigFromNode(node):
	"""given an ephRig node in the scene,
	return its python rig object"""
	obj = ephPyObject(node)
	return obj.ephRig






def connectedNodeObj(plug)->om.MObject:
	"""return single node MFns connected to this plug"""
	if plug.isSource:
		if plug.destinations():
			return plug.destinations()[0]
		return None
	elif plug.isDestination:
		return plug.source()
	return None




def reissue_uuids(request=None):
	"""
	Generates new UUIDs for the requested dependency nodes, including 'read-only' nodes.

	Request resolution order
		1. requested node or list of nodes
		2. selected nodes
		3. all dependency nodes loaded in scene

	:param request: Nodes to issue new UUIDs
	:type request: list|str|unicode

	:returns: Nothing
	:rtype: None
	"""
	get_depends = partial(cmds.ls, dep=True)
	depend_nodes = get_depends(request) or get_depends(selection=True) or get_depends()

	selection_list = om.MSelectionList()
	[selection_list.add(node) for node in depend_nodes]

	iter_selection = om.MItSelectionList(selection_list)  # MItDag unavailable until Maya 2016 SP6 ext 2
	while not iter_selection.isDone():
		node = om.MFnDependencyNode(iter_selection.getDependNode())
		uuid = om.MUuid().generate()
		node.setUuid(uuid)
		iter_selection.next()