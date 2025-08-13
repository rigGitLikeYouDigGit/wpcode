
from __future__ import annotations
import networkx as nx
import numpy as np
from dataclasses import dataclass
import typing as T

if T.TYPE_CHECKING:
	from edRig.maya.tool.feldspar.plate import Plate
	from edRig.maya.tool.feldspar.assembly import Assembly

@dataclass
class Vertex:
	"""single point on a plate, holding
	only its local position within a plate,
	and its parent plate"""

	plate:Plate=None
	index:int=0
	pos:np.ndarray=np.zeros(4)

	translator:np.ndarray=np.zeros(4)
	rotator:np.ndarray=np.zeros(4)

	def globalPos(self)->np.ndarray:
		return np.dot(self.plate.matrix.T,  self.pos)

	def globalRestPos(self)->np.ndarray:
		return np.dot(self.plate.restMatrix.T, self.pos)

	def reset(self):
		self.translator = np.zeros(4)
		self.rotator = np.zeros(4)
	# def __init__(self,
	#              plate:Plate=None,
	#              index:int=0,
	#              pos:np.ndarray=None):
	# 	self.plate = plate
	# 	self.index = index
	# 	self.pos = pos
	#
	# 	# sum of all translation vectors on this point
	# 	self.translator = np.zeros(3)
	#
	# 	# product of all rotators acting through this point
	# 	self.rotator = np.zeros(3)

	def __hash__(self):
		return id(self)





