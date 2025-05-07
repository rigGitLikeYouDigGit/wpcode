from __future__ import annotations
import typing as T

from ..gen.nurbsCurve import NurbsCurve as GenNurbsCurve
import numpy as np
from wpm import cmds, om, WN, to, arr

from wpm.lib import curve

if T.TYPE_CHECKING:
	from ...node.base import Plug

from ...node.base import filterToMObject, filterToMPlug, Plug, DGDagModifier

class NurbsCurve(GenNurbsCurve):
	""" moving nodes around in a more fluid way than walls
	of api calls"""
	MFn : om.MFnNurbsCurve
	clsApiType = om.MFn.kNurbsCurve

	kOpen = om.MFnNurbsCurve.kOpen
	kClosed = om.MFnNurbsCurve.kClosed

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

	# use create() to define order of arguments and process on creation,
	# WNMeta should delegate to this
	@classmethod
	def create(cls,
	           name:str="", dgMod_:DGDagModifier=None,
	           parent_:(str, om.MObject, WN)=None,
	           cvs=[om.MPoint(), om.MPoint()],
	           knots=None,
	           degree=2,
	           form=om.MFnNurbsCurve.kOpen,
	           is2d=False,
	           rational=True
	           )->WN:
		"""
		explicitly create a new node of this type, incrementing name if necessary

		suffix _ to avoid name clashes with kwargs

		leaving this as the single method to override for specific creation behaviour,
		without a deeper "_createInner()" method to actually create the object -
		will result in some copying of the argument processing here,
		but I find copying can sometimes be comforting for work code
		"""
		# creating from raw WN class, default to Transform
		if cls is WN:
			nodeCls = WN.Transform
		else:
			nodeCls = cls
		if parent_ is not None:
			parent_ = filterToMObject(parent_)
		opMod = dgMod_ or DGDagModifier()
		name = name or nodeCls.typeName + str(1)

		knots = knots or curve.knotArrayForCurve(len(cvs), degree)


		newObj = om.MFnNurbsCurve().create(
			cvs,
			knots,
			degree,
			form,
			is2d,
			rational,
			parent_ or om.MObject.kNullObj
		)
		# returns the created transform node, if you create a shape with no parent transform

		wrapper = nodeCls(newObj)
		wrapper.setName(name)

		return wrapper
