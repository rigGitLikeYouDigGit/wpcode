
from __future__ import annotations
import types, typing as T

from chimaera import ChimaeraNode

from wpm import WN, cmds, om, oma
from wpm.chimaera.scene import chiGrp, chiSel

"""
naming - would prefer to call this "MayaChiNode" and have a
further subclass of "RigOp", but I don't think it's that useful
"""
class MayaOp(ChimaeraNode):

	def rigGrp(self)->WN:
		parent = None
		if self.parent:
			parent= self.parent.rigGrp()
		return WN.create("transform", n=self.name + "_GRP",
		                 parent=parent,
		                 existOk=True)


