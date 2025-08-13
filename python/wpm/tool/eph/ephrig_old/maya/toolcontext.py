
"""MIGHT see if it's better to write this as a full context plugin
instead of just in cmds"""

from edRig import cmds, om

from edRig.maya.toolcontext import MayaToolContext

# class EphRigContext(MayaToolContext):
# 	"""allows user interaction with ephrig system"""
#
#
# 	def activate(self):
# 		print("activate ephrig ctx")
# 		super(EphRigContext, self).activate()
#
# 	def onDrag(self):
# 		print("eph drag")
# 		print("anchor", self.anchorPos, "drag", self.dragPos)