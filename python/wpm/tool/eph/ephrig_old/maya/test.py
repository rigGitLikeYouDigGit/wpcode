

import sys, os
import edRig

from edRig import ECA, cmds, om

ctx = None
def build():
	"""main eph test build script"""
	from edRig.ephrig import rig, node, solver
	cmds.unloadPlugin(ephName, f=1)

	for i in (rig, node, solver):
		#reload(i)
		pass
	from edRig.ephrig.maya import node as menode, pluginnode, lib, util, plugincontext
	for i in (menode, lib, util):
		#reload(i)
		pass
	cmds.loadPlugin(ephName)


	baseDir = os.path.join(*os.path.split(__file__)[:-1])
	plugincontext.updateMelFiles(
		baseDir, plugincontext.EphRigMpxAnimContext.kToolName)


	ctlNode = cmds.createNode("ephRigMain")
	print("create cmd", cmds.createNode)


	rootEph = menode.MEphNode.create("root", pos=(0, 0, 0))
	midEph = menode.MEphNode.create("midPoint", pos=(0, 5, 0))
	endEph = menode.MEphNode.create("endPoint", pos=(0, 10, 0))
	endEph.parentTf.set("translateX", 3)
	#
	# set up groups
	baseGrp = ECA("transform", "ephRig_grp")
	sourceGrp = ECA("transform", "source_grp", parent=baseGrp)
	outputGrp = ECA("transform", "output_grp", parent=baseGrp)

	for i, v in enumerate((rootEph, midEph, endEph)):
		cmds.parent(v.parentTf, sourceGrp)
		cmds.parent(v.outTf, outputGrp)
		v.connectToRigNode(ctlNode, index=i)
		v.outTf.setColour((1, 0, 0))
	outputGrp.set("translate", (1, 0, 0))

	# build rig on node
	newRig = util.buildRigFromNode(ctlNode)
	pyObj = lib.ephPyObject(ctlNode)
	#
	# # set connections in rig
	# print("nodes", newRig.nodes)
	newRig.addConnection(rootEph, midEph)
	newRig.addConnection(midEph, endEph)

	newRig.saveToNode()
	#
	# from maya import cmds as oldCmds
	# ctxCmd = getattr(oldCmds, plugincontext.EphRigContextSetupCmd.kPluginCmdName)
	# result = ctxCmd()
	# cmds.setToolTo(result)








ephName = "ephRigPlugin"
ephPath = "F:/all_projects_desktop/common/edCode/edRig/ephrig/maya"
def testMain():
	print("eph test")


	path = os.environ.get("MAYA_PLUG_IN_PATH", "")
	if not ephPath in path:
		path = ";" + ephPath + ";"
		os.environ["MAYA_PLUG_IN_PATH"] = path

	cmds.file(new=1, force=1)
	cmds.unloadPlugin(ephName)
	cmds.loadPlugin(ephName)

	build()
	# pass