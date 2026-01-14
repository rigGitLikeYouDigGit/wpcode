from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import cmds, om, oma, WN

from wpsim.kine import builder

"""maya-facing setup functions to build wpsim from maya nodes"""


#region body transforms
class MayaTransformSchema:
	"""test base class for expected structure
	under single transform"""

	def __init__(self, tfRoot:WN|str):
		self.tfRoot : WN.Transform = WN.Transform(tfRoot)



class WpSimRigidTfSchema(MayaTransformSchema):
	"""group representing single rigid body"""

	def name(self):
		return self.tfRoot.name()

	def comLoc(self)->WN.Transform:
		"""return locator representing body's centre of mass -
		will have a local offset compared to outer transform
		"""
		return self.tfRoot("")


class WpSimRigidGroup:
	"""simple system for now -
	prefix each body group with "b"
	"""
	def __init__(self, name="bFemur"):
		self.name = name

	def rootNodeName(self):
		return f"{self.name}_GRP"

	def exists(self):
		return cmds.objExists(self.rootNodeName())

	def tfRoot(self)->WN.Transform:
		"""root transform of group"""
		return WN(self.rootNodeName())

	def comLoc(self)->WN.Transform:
		"""return locator representing body's centre of mass"""
		return self.tfRoot()("com")

	def build(self):
		"""create new group, return top transform"""
		tf = WN.Transform(self.rootNodeName())
		comLoc = WN.Locator(f"{self.name}_com_LOC").tf()
		comLoc.setParentKeepWorld(tf)
