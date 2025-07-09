
from __future__ import annotations
import typing as T
from wplib import log

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
	pluginPath=str(WP_ROOT_PATH/"code"/"maya"/"plugin"/"strata"/"Debug"/"strata.mll"),
	#pluginPath=str(WP_ROOT_PATH/"code"/"maya"/"plugin"/"output"/"Maya2023"/"plug-ins"/"strata.mll"),
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
	# loc = cmds.createNode("strataPoint", n=name + "Shape")
	# tf = cmds.listRelatives(loc, parent=1)[0]
	# tf = cmds.rename(tf, name)
	# #cmds.connectAttr(loc + ".stFinalOutMatrix", tf + ".offsetParentMatrix", f=1)
	# return tf

	loc = WN.createNode("strataPoint", name)
	return loc

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


def reloadFrameCurveTest():
	from wpm import cmds
	cmds.file(newFile=1, f=1)
	pluginAid.loadPlugin(forceReload=True)
	# pt = cmds.createNode("strataPoint")

	# update node class wrappers
	pluginAid.updateGeneratedNodeClasses()

	refCurve = cmds.curve(point=[(0, 0, 0), (10, 0, 0)], degree=1)
	refCurve = WN(refCurve)
	refCurve.setName("ref_CRV")

	curve = cmds.curve(point=[(0, 1, 0), (1, 0, 0), (2, 3, 1), (4, -2, 1), (6, 1, -1), (10, 3, 3)])
	curve = WN(curve)
	curve.setName("active_CRV")
	curve.tf().translateY_ = 5
	print(curve.local_)

	frameCurve = WN.FrameCurve.create("frameCurve")

	frameCurve.refCurveIn_ = refCurve.worldSpace_[0]
	frameCurve.curveIn_ = curve.worldSpace_[0]

	tfs = []
	for i in range(0, 7):
		tf = WN.Locator.create("refTf{}_LOC".format(i)).tf()
		tf.translateX_ = i + 1
		tf.translateZ_ = 1
		tfs.append(tf)

		frameCurve.sampleIn_[i].refSampleMatrixIn_ = tf.worldMatrix_[0]

		outTf = WN.Locator.create("outTf{}_LOC".format(i)).tf()
		outTf.offsetParentMatrix_ = frameCurve.sampleOut_[i].sampleMatrixOut_

	# up locators
	refUpTfA = WN.Locator.create("refUpTfStart_LOC").tf()
	refUpTfA.translateX_ = -3

	refUpTfB = WN.Locator.create("refUpTfEnd_LOC").tf()
	refUpTfB.translateX_ = 13

	activeUpTfA = WN.Locator.create("activeUpTfStart_LOC").tf()
	activeUpTfA.translateY_ = 5

	activeUpTfB = WN.Locator.create("activeUpTfEnd_LOC").tf()
	activeUpTfB.translateX_ = 13
	activeUpTfB.translateY_ = 5

	frameCurve.refUp_[0].refUpMatrix_ = refUpTfA.worldOut
	frameCurve.refUp_[1].refUpMatrix_ = refUpTfB.worldOut

	frameCurve.activeUp_[0].activeUpMatrix_ = activeUpTfA.worldOut
	frameCurve.activeUp_[1].activeUpMatrix_ = activeUpTfB.worldOut
	return


def reloadPluginTest():
	"""single test to check strata nodes are working

	from wpm.tool.strata import plugin
	plugin.reloadPluginTest()
	"""
	from wpm import cmds
	cmds.file(newFile=1, f=1)
	pluginAid.loadPlugin(forceReload=True)
	#pt = cmds.createNode("strataPoint")

	# update node class wrappers
	pluginAid.updateGeneratedNodeClasses()




	# create single strata shape
	elOp: WN.StrataElementOp = WN.createNode("strataElementOp", "newElOp")
	elOp.stElement_[0].stName_ = "pShoulder"
	ptA = makeStrataPoint("ptA").tf()


	elOp.stElement_[0].stPointWorldMatrixIn_ = ptA.worldMatrix_[0]
	#return
	log(elOp.stElement_, elOp, type(elOp))
	# connect up element to shape
	shape : WN.StrataShape = WN.StrataShape.create("newShape")
	shape.stInput_[0] = elOp.stOutput_

	# el2 = WN.createNode("strataElementOp", "el2Op")
	# shape.stInput_[1] = el2.stOutput_
	# shape.stInput_[2] = elOp.stOutput_

	#return


	ptB = makeStrataPoint("ptB").tf()
	ptB.translateX_ = 3

	ptC = makeStrataPoint("ptC").tf()
	ptC.translateX_ = 6


	elOp.stElement_[1].stName_ = "pElbow"
	elOp.stElement_[1].stPointWorldMatrixIn_ = ptB.worldMatrix_[0]
	# set parents
	elOp.stElement_[1].stSpaceExp = "pShoulder"

	elOp.stElement_[2].stName_ = "pWrist"
	elOp.stElement_[2].stPointWorldMatrixIn_ = ptC.worldMatrix_[0]
	elOp.stElement_[2].stSpaceExp_ = "pElbow"


	# connect elbow driver to test space driving, and FK
	driverLoc = WN.Locator.create("elbowDriver_LOC")
	# shape.stDataIn_[0].stExpIn_ = "pWrist"
	# shape.stDataIn_[0].stSpaceNameIn_ = "pElbow"
	# shape.stDataIn_[0].stMatrixIn_ = driverLoc.matrix_

	shape.stDataIn_[0].stExpIn_ = "pElbow"
	shape.stDataIn_[0].stMatrixIn_ = driverLoc.matrix_

	cmds.setAttr(driverLoc + ".rotateZ", 45)

	return


	#elOp = cmds.createNode("strataElementOp")















