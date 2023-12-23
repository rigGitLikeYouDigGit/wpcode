

from chimaera import ChimaeraNode

from wpm import WN, cmds, om, oma


class MayaOp(ChimaeraNode):

	def rigGrp(self)->WN:
		parent = None
		if self.parent():
			parent= self.parent().rigGrp()
		return WN.create("transform", n=self.name + "_GRP",
		                 parent=parent,
		                 existOk=True)


