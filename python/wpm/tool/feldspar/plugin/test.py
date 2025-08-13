from maya import cmds
from edRig import ECA, EdNode
from edRig.maya.object import PlugTree

"""create a single 4-bar linkage
if we can't do this we fail"""

def setBarTransforms(node:EdNode, barIndex, aIndex, bIndex):
	""""""
	node("bar", barIndex, "vertex0").set(aIndex)
	node("bar", barIndex, "vertex1").set(bIndex)


def test_simple():
	baseA = ECA("locator", n="baseA")
	rockA = ECA("locator", n="rockA")

	# output vertex for tracing
	trace = ECA("locator", n="trace")

	# output for targeting
	target = ECA("locator", n="target")

	baseA("translate").set(0, 0, 0)

	rockA("translate").set(0, 2, 1)
	trace("translate").set(0, 3, 0)
	target("translate").set(0, 4, 0)

	setupNode = ECA("feldsparSetup")
	baseA("translate").con(setupNode("vertex", 0, "pos"))
	rockA("translate").con(setupNode("vertex", 1, "pos"))
	trace("translate").con(setupNode("vertex", 2, "pos"))
	target("translate").con(setupNode("vertex", 3, "pos"))

	setBarTransforms(setupNode, 0, 0, 1)
	setBarTransforms(setupNode, 1, 1, 2)
	setBarTransforms(setupNode, 2, 2, 3)
	setupNode("bar", 2, "soft").set(True)

	# set rigid group
	setupNode("group", 0, "groupVertexIndex").set(0, 3)
	print("vertex index", setupNode("group", 0, "groupVertexIndex").get())
	setupNode("group", 0, "groupVertexIndex", 0).set(0)
	setupNode("group", 0, "groupVertexIndex", 1).set(3)
	print("vertex index", setupNode("group", 0, "groupVertexIndex").get())
	setupNode("group", 0, "groupFixed").set(True)

	cmds.setAttr(setupNode + ".bind", 1)

	# setup solver
	solverNode = ECA("feldsparSolver" )

	setupNode("tick").con(solverNode("graphTick"))

	# outputs
	baseAJnt = ECA("joint", "baseA_jnt")
	rockAJnt = ECA("joint", "rockA_jnt")
	traceJnt = ECA("joint", "trace_jnt")
	targetJnt = ECA("joint", "target_jnt")
	for i, v in enumerate((baseAJnt, rockAJnt, traceJnt, targetJnt)):
		solverNode("graphOutput", 0, "vertex", i, "pos").con(v.translate)


def test_deadSimple():
	"""Single link, single target, straight line"""
	baseA = ECA("locator", n="baseA")
	rockA = ECA("locator", n="rockA")

	# output vertex for tracing
	#trace = ECA("locator", n="trace")

	# output for targeting
	target = ECA("locator", n="target")

	baseA("translate").set(0, 0, 0)

	rockA("translate").set(0, 2, 2)
	#trace("translate").set(0, 3, 0)
	target("translate").set(0, 4, 0)

	setupNode = ECA("feldsparSetup")
	baseA("translate").con(setupNode("vertex", 0, "pos"))
	rockA("translate").con(setupNode("vertex", 1, "pos"))
	#trace("translate").con(setupNode("vertex", 2, "pos"))
	target("translate").con(setupNode("vertex+", "pos"))

	setBarTransforms(setupNode, 0, 0, 1)
	setBarTransforms(setupNode, 1, 1, 2)
	#setBarTransforms(setupNode, 2, 2, 3)
	setupNode("bar", 1, "soft").set(True)

	# set rigid group
	#setupNode("group", 0, "groupVertexIndex").set(0, 2)
	#print("vertex index", setupNode("group", 0, "groupVertexIndex").get())
	setupNode("group", 0, "groupVertexIndex", 0).set(0)
	setupNode("group", 0, "groupVertexIndex", 1).set(2)
	#print("vertex index", setupNode("group", 0, "groupVertexIndex").get())
	setupNode("group", 0, "groupFixed").set(True)

	cmds.setAttr(setupNode + ".bind", 1)

	# setup solver
	solverNode = ECA("feldsparSolver" )

	setupNode("tick").con(solverNode("graphTick"))

	# outputs
	baseAJnt = ECA("joint", "baseA_jnt")
	rockAJnt = ECA("joint", "rockA_jnt")
	#traceJnt = ECA("joint", "trace_jnt")
	targetJnt = ECA("joint", "target_jnt")
	for i, v in enumerate((baseAJnt, rockAJnt, #traceJnt,
	                       targetJnt)):
		solverNode("graphOutput", 0, "vertex", i, "pos").con(v.translate)

def addNewBar(node:EdNode, idA, idB):
	plug = node("bar+")
	plug("vertex0").set(idA)
	plug("vertex1").set(idB)

def runTest():
	#return test_simple()
	#return test_deadSimple()
	#return test_mid()

	# work in yz plane

	# create vertex points
	baseA = ECA("locator", n="baseA")
	baseB = ECA("locator", n="baseB")

	rockA = ECA("locator", n="rockA")
	rockB = ECA("locator", n="rockB")

	# output vertex for tracing
	trace = ECA("locator", n="trace")

	# output for targeting
	target = ECA("locator", n="target")

	baseA("translate").set(0, 0, 2)
	baseB("translate").set(0, 0, -1)

	rockA("translate").set(0, 2, 2)
	rockB("translate").set(0, 3, -1)

	trace("translate").set(0, 4, 0)
	target("translate").set(0, 4, 1)

	locs = (baseA, baseB, rockA, rockB, trace, target)
	# create plates
	setupNode = ECA("feldsparSetup")
	for i, loc in enumerate(locs):
		loc.translate.con(setupNode("vertex", i, "pos"))

	# set bars
	addNewBar(setupNode, 0, 2)
	addNewBar(setupNode, 1, 3)
	addNewBar(setupNode, 2, 3)
	addNewBar(setupNode, 2, 4)
	addNewBar(setupNode, 3, 4)
	addNewBar(setupNode, 4, 5)
	setupNode("bar", -1, "soft").set(True)

	# add fixed group
	setupNode("group", 0, "groupVertexIndex+").set(0)
	setupNode("group", 0, "groupVertexIndex+").set(1)
	setupNode("group", 0, "groupVertexIndex+").set(5)
	setupNode("group", 0, "groupFixed").set(True)

	cmds.setAttr(setupNode + ".bind", 1)


	# connect all nodes to solver
	solver = ECA("feldsparSolver", n="solver").shape
	setupNode("tick").con(solver("graphTick"))

	for i, loc in enumerate(locs):
		jnt = ECA("joint", n=loc.name + "_jnt")
		solver("graphOutput", 0, "vertex", i, "pos").con(jnt.translate)

	cmds.setAttr(solver + ".bind", 1)

	# cubeMap = {}
	#
	# for i, plate in enumerate((fixPlate, targetPlate, rockAPlate, rockBPlate, rockMidPlate
	#               )):
	# 	plate("tick").con(solver("plateInput+", "plateTick"))
	# 	outCube = EdNode(cmds.polyCube(n=plate.name + "_outCube")[0])
	# 	outCube("scale").set(0.2, 1.0, 0.5)
	#
	# 	cubeMap[plate] = outCube
	# 	solver("plateOutput", i, "matrix").con(outCube.offsetParentMatrix)
	#
	#
	# for constraint in (consFixA,
	#                    consFixB, consMidA, consMidB,
	# 	consTarget
	#                    ):
	# 	constraint("tick").con(solver("constraintInput+", "constraintTick"))
	#
	#
	# # cmds to let the program catch up before running
	# cmds.setAttr(solver + ".bind", 1)
	# solver("maxIterations").set(3)
	#
	# for i, plate in enumerate((fixPlate, targetPlate, rockAPlate, rockBPlate, rockMidPlate
	#               )):
	# 	outCube = cubeMap[plate]
	# 	# draw joints
	# 	locs = []
	# 	for driverPlug, drivenPlug in plate("vertex").drivers(allSubPlugs=True):
	# 		locs.append(EdNode(driverPlug.node).transform)
	#
	# 	for n, loc in enumerate(locs):
	# 		prev = n - 1
	# 		jntA = ECA("joint", n=plate.name + "{}_jointA".format(n))
	# 		jntB = ECA("joint", n=plate.name + "{}_jointB".format(n))
	# 		jntA.translate.set(locs[prev].translate.get())
	# 		jntA("radius").set(0.3)
	# 		jntB.translate.set(loc.translate.get())
	# 		jntB("radius").set(0.3)
	# 		jntB.parentTo(jntA)
	# 		jntA.parentTo(outCube)


"""
from edRig.maya.tool.feldspar.plugin import loadPlugin
loadPlugin()
from edRig.maya.tool.feldspar.plugin.test import runTest
runTest()
"""