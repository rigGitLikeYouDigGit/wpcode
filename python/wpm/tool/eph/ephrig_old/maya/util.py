from edRig import cmds, om

#from edRig.ephrig.rig import EphRig, EphSolver
from edRig.ephrig.maya.rig import MEphRig
from edRig.ephrig.maya.node import MEphNode
from edRig.ephrig.maya.pluginnode import EphRigControlNode
# from edRig.ephrig.maya.plugincontext import EphRigMpxAnimContext
# context depends on utils
from edRig.ephrig.maya import lib

from edRig.maya.lib import attr

reload(attr)




def ephNodeFromEntry(ctlNode, index):
	"""given node and index, return
	new EphNode object initialised with those values"""
	baseInPlug = ctlNode + ".inputArray[{}]".format(index)
	baseOutPlug = ctlNode + ".outputArray[{}]".format(index)
	childPlug = attr.getMPlug(baseInPlug + ".inChildMsg")
	child = lib.connectedNodeObj(childPlug)
	parentPlug = attr.getMPlug(baseInPlug + ".inParentMsg")
	parent = lib.connectedNodeObj(parentPlug)
	outPlug = attr.getMPlug(baseOutPlug + ".outWorldMat")
	output = lib.connectedNodeObj(outPlug)
	namePlug = attr.getMPlug(baseInPlug + ".nodeName")
	name = namePlug.asString()

	eNode = MEphNode(name, parent, child, output)
	return eNode



def getRigFromNode(node:T.Union[EphRigControlNode, str, om.MObject]):
	"""returns a node's live rig if one is already attached"""
	node = lib.ephPyObject(node)
	return node.ephRig

def buildRigFromNode(node, attach=True):
	"""given ephControlNode, build eph rig from its
	current connections
	creates and links EphNodes
	"""
	nAttached = cmds.getAttr(node + ".inputArray", size=1)
	newRig = MEphRig()
	for i in range(nAttached):
		eNode = ephNodeFromEntry(node, i)
		newRig.addNode(eNode)

	if attach: # attach rig to control node
		pyObj = lib.ephPyObject(node)
		pyObj.setRig(newRig)

	newRig.setMainNode(node)

	return newRig




