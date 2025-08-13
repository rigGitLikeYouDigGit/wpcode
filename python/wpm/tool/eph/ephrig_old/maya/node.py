
"""naming scheme already getting a bit dicey"""

from edRig import ECA, EdNode, con
from edRig.maya.lib import attr
from edRig.ephrig.node import EphNode


class MEphNode(EphNode):
	"""
	maya-facing eph node facade
	in interest of making this as usable as possible,
	maybe don't play with ed-types?
	"""

	def __init__(self,
	             name,
	             inParentTf,
	             inChildTf,
	             outTf
	             ):
		super(MEphNode, self).__init__(name)
		self.parentTf = EdNode(inParentTf)
		self.childTf = EdNode(inChildTf)
		self.outTf = EdNode(outTf)




	def connectToRigNode(self, rigNode, index=0):
		"""connects this ephNode to a central maya
		 controller node"""
		basePlug = rigNode + ".inputArray[{}]".format(index)

		con(self.parentTf + ".worldMatrix", basePlug + ".inParentMat")
		con(self.childTf + ".worldMatrix", basePlug + ".inWorldMat")
		outPlug = rigNode + ".outputArray[{}]".format(index)
		con(outPlug + ".outWorldMat", self.outTf + ".offsetParentMatrix")

		con(self.parentTf + ".message", basePlug + ".inParentMsg")
		con(self.childTf + ".message", basePlug + ".inChildMsg")

		namePlug = attr.getMPlug(basePlug + ".nodeName")
		namePlug.setString(self.name)


		# attach directly to the node's python rig object???



	@classmethod
	def create(cls, name, pos=(0, 0, 0)):
		"""create the trio of joints, link up message attrs,
		return initialised EphNode"""
		parent = ECA("joint", name=name + "Parent_ephJnt")
		child = ECA("joint", name=name + "_ephJnt", parent=parent)
		output = ECA("joint", name=name + "Output_ephJnt")

		parent.addAttr("child", type="message")
		parent.addAttr("output", type="message")
		child.addAttr("parent", type="message")
		child.addAttr("output", type="message")
		output.addAttr("parent", type="message")
		output.addAttr("child", type="message")

		parent("child").con(child("parent"))
		parent("output").con(output("parent"))
		child("output").con(output("child"))
		parent("translate").set(pos)

		# for i in (parent, output):
		# 	#i.set("translate", pos)
		# 	i("translate").set(pos)

		meNode = MEphNode(name, parent(), child(), output())
		return meNode



