from __future__ import annotations
import typing as T

from ..gen.nurbsCurve import NurbsCurve as GenNurbsCurve
import numpy as np
from wpm import cmds, om, WN, to, arr

if T.TYPE_CHECKING:
	from ...node.base import Plug


class NurbsCurve(GenNurbsCurve):
	""" moving nodes around in a more fluid way than walls
	of api calls"""
	MFn : om.MFnNurbsCurve
	clsApiType = om.MFn.kNurbsCurve


	@property
	def localIn(self) -> Plug:
		return self.create_
	@property
	def localOut(self) -> Plug:
		return self.local_

	@property
	def worldIn(self) -> Plug:
		return self.create_
	@property
	def worldOut(self) -> Plug:
		return self.worldSpace_[0]