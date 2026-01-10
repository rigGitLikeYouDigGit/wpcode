from __future__ import annotations
import typing as T

from ..gen.mesh import Mesh as GenMesh
import numpy as np
from wpm import cmds, om, WN, arr
from wplib.totype import to

if T.TYPE_CHECKING:
	from ...node.base import Plug


class Mesh(GenMesh):
	MFn : om.MFnMesh
	clsApiType = om.MFn.kMesh


	@property
	def localIn(self) -> Plug:
		return self.inMesh_
	@property
	def localOut(self) -> Plug:
		return self.outMesh_

	@property
	def worldIn(self) -> Plug:
		return self.inMesh_
	@property
	def worldOut(self) -> Plug:
		return self.worldMesh_[0]