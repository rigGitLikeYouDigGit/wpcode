
from __future__ import annotations
import typing as T


"""
"""

from wplib import WP_ROOT_PATH
from wpm import WN
from wpm.lib.plugin import PluginNodeTemplate, WpPluginAid

from wpm.tool.strata.plugin.point import StrataPoint

from wpm.tool import strata

from pathlib import Path
thisFilePath = Path(strata.__file__).parent / "plugin" / "__init__.py"

# class StrataPluginAid(MayaPluginAid):
# 	@classmethod
# 	def studioIdPrefix(cls)->int:
# 		"""return the prefix for studio plugin ids,
# 		used as a namespace for plugin node ids"""
# 		return WP_PLUGIN_NAMESPACE

#F:\wp\code\maya\plugin\strata\Debug
# construct overall plugin object, register any python plugin nodes
pluginAid = WpPluginAid(
	"strata",
	#pluginPath=str(WP_ROOT_PATH/"code"/"maya"/"plugin"/"strata"/"Debug"/"strata.mll"),
	pluginPath=str(WP_ROOT_PATH/"code"/"maya"/"plugin"/"output"/"Maya2023"/"plug-ins"/"strata.mll"),
	# nodeClasses={
	# 	1 : StrataPoint
	# }
)


def maya_useNewAPI():
	pass


def initializePlugin(plugin):
	"""initialise the plugin"""
	pluginAid.initialisePlugin(plugin)

def uninitializePlugin(plugin):
	"""uninitialise the plugin"""
	pluginAid.uninitialisePlugin(plugin)

def makeStrataPoint(name):
	"""I tried a little to do all this automatically when the node is created,
	but since we're forced to use a shape node, it's always gonna be
	a mess"""
	from maya import cmds
	loc = cmds.createNode("strataPoint", n=name + "Shape")
	tf = cmds.listRelatives(loc, parent=1)[0]
	tf = cmds.rename(tf, name)
	#cmds.connectAttr(loc + ".stFinalOutMatrix", tf + ".offsetParentMatrix", f=1)
	return tf

def connectStrataPoint(pointNode, addPointsNode):
	from maya import cmds
	nEntries = cmds.getAttr(addPointsNode + ".stParam", size=1)
	arrPlug = addPointsNode + ".stParam[{}]".format(nEntries)
	cmds.connectAttr(pointNode + ".stName", arrPlug + ".stParamExp")
	cmds.connectAttr(pointNode + ".worldMatrix[0]", arrPlug + ".stParamMat")

def makeStrataCurve(name, tf1, tf2):
	from maya import cmds
	dgNode = cmds.createNode("strataCurve", name= name+"Node")
	crv = cmds.createNode("nurbsCurve", name=name + "Shape")
	tf = cmds.rename(cmds.listRelatives(crv, parent=1)[0], name)
	cmds.connectAttr(dgNode + ".stOutCurve", crv + ".create", f=1)
	cmds.connectAttr(tf1 + ".worldMatrix[0]", dgNode + ".stStartMatrix")
	cmds.connectAttr(tf2 + ".worldMatrix[0]", dgNode + ".stEndMatrix")
	return dgNode, tf

def reloadPluginTest():
	"""single test to check strata nodes are working"""
	from maya import cmds
	cmds.file(newFile=1, f=1)
	pluginAid.loadPlugin(forceReload=True)
	#pt = cmds.createNode("strataPoint")

	# test creating a single face of strataSurface
	ptA = makeStrataPoint("ptA")
	cmds.setAttr(ptA + ".translate", -3, 2, -3)

	ptB = makeStrataPoint("strataPointB")
	ptC = makeStrataPoint("strataPointC")
	ptD = makeStrataPoint("strataPointD")
	# # move points
	cmds.setAttr(ptA + ".translate", -3, 2, -3)
	cmds.setAttr(ptB + ".translate", 2, 3, -3)
	#
	cmds.setAttr(ptC + ".translate", 2, -1, -3)
	cmds.setAttr(ptD + ".translate", 3, -1, -3)

	#return
	#elOp = cmds.createNode("strataElementOp")
	elOp = WN.createNode("strataElementOp", "newElOp")
	elOp.stElement_[0].stName_ = "pTopPt"
	elOp.stElement_[1].stName_ = "pMidPt"
	# elOp.stElement_[2].stName_ = "pLowPt"
	# elOp.stElement_[3].stName_ = "eCentre"
	# elOp.stElement_[3].stExp_ = "pTopPt, pMidPt, pLowPt"


	return
	#graphNode = cmds.createNode("strataGraph")

	# addPointsNode = cmds.createNode("strataAddPointsOp")
	# cmds.connectAttr(graphNode + ".stGraph", addPointsNode + ".stGraph")
	# addPointsNode2 = cmds.createNode("strataAddPointsOp")
	# cmds.connectAttr(graphNode + ".stGraph", addPointsNode2 + ".stGraph")
	# addPointsNode3 = cmds.createNode("strataAddPointsOp")
	# cmds.connectAttr(graphNode + ".stGraph", addPointsNode3 + ".stGraph")

	return
	dgNode, crvTf = makeStrataCurve("strataCurveA", ptA, ptB)

	dgNodeB, crvTf = makeStrataCurve("strataCurveB", ptC, ptD)
	#












