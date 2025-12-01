

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
AbstractBaseNurbsConversion = retriever.getNodeCls("AbstractBaseNurbsConversion")
assert AbstractBaseNurbsConversion
if T.TYPE_CHECKING:
	from .. import AbstractBaseNurbsConversion

# add node doc



# region plug type defs
class ChordHeightPlug(Plug):
	node : ParentTessellate = None
	pass
class ChordHeightRatioPlug(Plug):
	node : ParentTessellate = None
	pass
class DeltaPlug(Plug):
	node : ParentTessellate = None
	pass
class EdgeSwapPlug(Plug):
	node : ParentTessellate = None
	pass
class FormatPlug(Plug):
	node : ParentTessellate = None
	pass
class FractionalTolerancePlug(Plug):
	node : ParentTessellate = None
	pass
class MatchNormalDirPlug(Plug):
	node : ParentTessellate = None
	pass
class MinEdgeLengthPlug(Plug):
	node : ParentTessellate = None
	pass
class NormalizeTrimmedUVRangePlug(Plug):
	node : ParentTessellate = None
	pass
class OutputPolygonPlug(Plug):
	node : ParentTessellate = None
	pass
class PolygonCountPlug(Plug):
	node : ParentTessellate = None
	pass
class PolygonTypePlug(Plug):
	node : ParentTessellate = None
	pass
class Pre70ChordHeightRatioPlug(Plug):
	node : ParentTessellate = None
	pass
class UNumberPlug(Plug):
	node : ParentTessellate = None
	pass
class UTypePlug(Plug):
	node : ParentTessellate = None
	pass
class UseChordHeightPlug(Plug):
	node : ParentTessellate = None
	pass
class UseChordHeightRatioPlug(Plug):
	node : ParentTessellate = None
	pass
class VNumberPlug(Plug):
	node : ParentTessellate = None
	pass
class VTypePlug(Plug):
	node : ParentTessellate = None
	pass
# endregion


# define node class
class ParentTessellate(AbstractBaseNurbsConversion):
	chordHeight_ : ChordHeightPlug = PlugDescriptor("chordHeight")
	chordHeightRatio_ : ChordHeightRatioPlug = PlugDescriptor("chordHeightRatio")
	delta_ : DeltaPlug = PlugDescriptor("delta")
	edgeSwap_ : EdgeSwapPlug = PlugDescriptor("edgeSwap")
	format_ : FormatPlug = PlugDescriptor("format")
	fractionalTolerance_ : FractionalTolerancePlug = PlugDescriptor("fractionalTolerance")
	matchNormalDir_ : MatchNormalDirPlug = PlugDescriptor("matchNormalDir")
	minEdgeLength_ : MinEdgeLengthPlug = PlugDescriptor("minEdgeLength")
	normalizeTrimmedUVRange_ : NormalizeTrimmedUVRangePlug = PlugDescriptor("normalizeTrimmedUVRange")
	outputPolygon_ : OutputPolygonPlug = PlugDescriptor("outputPolygon")
	polygonCount_ : PolygonCountPlug = PlugDescriptor("polygonCount")
	polygonType_ : PolygonTypePlug = PlugDescriptor("polygonType")
	pre70ChordHeightRatio_ : Pre70ChordHeightRatioPlug = PlugDescriptor("pre70ChordHeightRatio")
	uNumber_ : UNumberPlug = PlugDescriptor("uNumber")
	uType_ : UTypePlug = PlugDescriptor("uType")
	useChordHeight_ : UseChordHeightPlug = PlugDescriptor("useChordHeight")
	useChordHeightRatio_ : UseChordHeightRatioPlug = PlugDescriptor("useChordHeightRatio")
	vNumber_ : VNumberPlug = PlugDescriptor("vNumber")
	vType_ : VTypePlug = PlugDescriptor("vType")

	# node attributes

	typeName = "parentTessellate"
	typeIdInt = 1346458707
	nodeLeafClassAttrs = ["chordHeight", "chordHeightRatio", "delta", "edgeSwap", "format", "fractionalTolerance", "matchNormalDir", "minEdgeLength", "normalizeTrimmedUVRange", "outputPolygon", "polygonCount", "polygonType", "pre70ChordHeightRatio", "uNumber", "uType", "useChordHeight", "useChordHeightRatio", "vNumber", "vType"]
	nodeLeafPlugs = ["chordHeight", "chordHeightRatio", "delta", "edgeSwap", "format", "fractionalTolerance", "matchNormalDir", "minEdgeLength", "normalizeTrimmedUVRange", "outputPolygon", "polygonCount", "polygonType", "pre70ChordHeightRatio", "uNumber", "uType", "useChordHeight", "useChordHeightRatio", "vNumber", "vType"]
	pass

