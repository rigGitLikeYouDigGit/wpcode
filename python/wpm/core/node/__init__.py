

from __future__ import annotations


#print("node import cmds", cmds)

from .main import WN
from .objectset import ObjectSetNode
from .remapvalue import RemapValue


"""package for defining custom wrappers around individual maya node types
many small files are better than a few big ones"""

from ..cache import cmds, om, oma

def createWN(nodeType, name="", n="", parent=None, existOk=False,
             returnTransform=True)->WN:
	"""create node and wrap as EdNode
	if findExisting, existing node of same name will be returned,
	else name will be incremented as per normal maya
	operation

	if returnTransform, if the raw node creation would return a shape
	eg locator, nurbsCurve, then return that shape's transform
	"""
	name = name or n or nodeType

	if existOk:
		if cmds.ls(name):
			return WN(name)


	# check if specific node wrapper class exists for nodeType
	wrapCls = WN.wrapperClassForNodeType(nodeType)

	if wrapCls is WN:
		edNode = WN(cmds.createNode(nodeType, n=name, parent=parent))
	else: # custom subclass
		print("calling create on", wrapCls)
		edNode = wrapCls.create(n=name, parent=parent)

	# avoid annoying material errors
	if nodeType == "mesh" or nodeType == "nurbsSurface":
		edNode.assignMaterial("lambert1")

	if edNode.isShape() and returnTransform:
		# set naming correctly
		edNode.parent.rename(name)
		return edNode.parent

	return edNode



def invokeNode(name="", nodeType="", parent=""):
	# print "core invokeNode looking for {}".format(name)
	return createWN(nodeType, name=name, parent=parent, existOk=True)




