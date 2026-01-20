from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from wpm import cmds, om, oma, WN

from wpsim.kine import builder

"""maya-facing setup functions to build wpsim from maya nodes"""



def getRigidTFSchema(
		name:str
)->WN.NodeSchema:
	"""return schema for rigid body with given name"""
	return WN.NodeSchema.fromLiteral(
		{f"{name}_GRP": (WN.T(), {
			"com": WN.Locator,
			"mesh": WN.Mesh,
		})}
	)



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
