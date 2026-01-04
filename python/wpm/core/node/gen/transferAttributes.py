

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	WeightGeometryFilter = Catalogue.WeightGeometryFilter
else:
	from .. import retriever
	WeightGeometryFilter = retriever.getNodeCls("WeightGeometryFilter")
	assert WeightGeometryFilter

# add node doc



# region plug type defs
class ColorBordersPlug(Plug):
	node : TransferAttributes = None
	pass
class FlipUVsPlug(Plug):
	node : TransferAttributes = None
	pass
class MatchChoicePlug(Plug):
	node : TransferAttributes = None
	pass
class MatchCountPlug(Plug):
	node : TransferAttributes = None
	pass
class SampleSpacePlug(Plug):
	node : TransferAttributes = None
	pass
class SearchDistancePlug(Plug):
	node : TransferAttributes = None
	pass
class SearchMethodPlug(Plug):
	node : TransferAttributes = None
	pass
class SearchScaleXPlug(Plug):
	parent : SearchScalePlug = PlugDescriptor("searchScale")
	node : TransferAttributes = None
	pass
class SearchScaleYPlug(Plug):
	parent : SearchScalePlug = PlugDescriptor("searchScale")
	node : TransferAttributes = None
	pass
class SearchScaleZPlug(Plug):
	parent : SearchScalePlug = PlugDescriptor("searchScale")
	node : TransferAttributes = None
	pass
class SearchScalePlug(Plug):
	searchScaleX_ : SearchScaleXPlug = PlugDescriptor("searchScaleX")
	ssx_ : SearchScaleXPlug = PlugDescriptor("searchScaleX")
	searchScaleY_ : SearchScaleYPlug = PlugDescriptor("searchScaleY")
	ssy_ : SearchScaleYPlug = PlugDescriptor("searchScaleY")
	searchScaleZ_ : SearchScaleZPlug = PlugDescriptor("searchScaleZ")
	ssz_ : SearchScaleZPlug = PlugDescriptor("searchScaleZ")
	node : TransferAttributes = None
	pass
class SearchTolerancePlug(Plug):
	node : TransferAttributes = None
	pass
class SourcePlug(Plug):
	node : TransferAttributes = None
	pass
class SourceColorSetPlug(Plug):
	node : TransferAttributes = None
	pass
class SourceUVSetPlug(Plug):
	node : TransferAttributes = None
	pass
class SourceUVSpacePlug(Plug):
	node : TransferAttributes = None
	pass
class TargetColorSetPlug(Plug):
	node : TransferAttributes = None
	pass
class TargetUVSetPlug(Plug):
	node : TransferAttributes = None
	pass
class TargetUVSpacePlug(Plug):
	node : TransferAttributes = None
	pass
class TransferColorsPlug(Plug):
	node : TransferAttributes = None
	pass
class TransferNormalsPlug(Plug):
	node : TransferAttributes = None
	pass
class TransferPositionsPlug(Plug):
	node : TransferAttributes = None
	pass
class TransferUVsPlug(Plug):
	node : TransferAttributes = None
	pass
# endregion


# define node class
class TransferAttributes(WeightGeometryFilter):
	colorBorders_ : ColorBordersPlug = PlugDescriptor("colorBorders")
	flipUVs_ : FlipUVsPlug = PlugDescriptor("flipUVs")
	matchChoice_ : MatchChoicePlug = PlugDescriptor("matchChoice")
	matchCount_ : MatchCountPlug = PlugDescriptor("matchCount")
	sampleSpace_ : SampleSpacePlug = PlugDescriptor("sampleSpace")
	searchDistance_ : SearchDistancePlug = PlugDescriptor("searchDistance")
	searchMethod_ : SearchMethodPlug = PlugDescriptor("searchMethod")
	searchScaleX_ : SearchScaleXPlug = PlugDescriptor("searchScaleX")
	searchScaleY_ : SearchScaleYPlug = PlugDescriptor("searchScaleY")
	searchScaleZ_ : SearchScaleZPlug = PlugDescriptor("searchScaleZ")
	searchScale_ : SearchScalePlug = PlugDescriptor("searchScale")
	searchTolerance_ : SearchTolerancePlug = PlugDescriptor("searchTolerance")
	source_ : SourcePlug = PlugDescriptor("source")
	sourceColorSet_ : SourceColorSetPlug = PlugDescriptor("sourceColorSet")
	sourceUVSet_ : SourceUVSetPlug = PlugDescriptor("sourceUVSet")
	sourceUVSpace_ : SourceUVSpacePlug = PlugDescriptor("sourceUVSpace")
	targetColorSet_ : TargetColorSetPlug = PlugDescriptor("targetColorSet")
	targetUVSet_ : TargetUVSetPlug = PlugDescriptor("targetUVSet")
	targetUVSpace_ : TargetUVSpacePlug = PlugDescriptor("targetUVSpace")
	transferColors_ : TransferColorsPlug = PlugDescriptor("transferColors")
	transferNormals_ : TransferNormalsPlug = PlugDescriptor("transferNormals")
	transferPositions_ : TransferPositionsPlug = PlugDescriptor("transferPositions")
	transferUVs_ : TransferUVsPlug = PlugDescriptor("transferUVs")

	# node attributes

	typeName = "transferAttributes"
	apiTypeInt = 992
	apiTypeStr = "kTransferAttributes"
	typeIdInt = 1414676820
	MFnCls = om.MFnGeometryFilter
	nodeLeafClassAttrs = ["colorBorders", "flipUVs", "matchChoice", "matchCount", "sampleSpace", "searchDistance", "searchMethod", "searchScaleX", "searchScaleY", "searchScaleZ", "searchScale", "searchTolerance", "source", "sourceColorSet", "sourceUVSet", "sourceUVSpace", "targetColorSet", "targetUVSet", "targetUVSpace", "transferColors", "transferNormals", "transferPositions", "transferUVs"]
	nodeLeafPlugs = ["colorBorders", "flipUVs", "matchChoice", "matchCount", "sampleSpace", "searchDistance", "searchMethod", "searchScale", "searchTolerance", "source", "sourceColorSet", "sourceUVSet", "sourceUVSpace", "targetColorSet", "targetUVSet", "targetUVSpace", "transferColors", "transferNormals", "transferPositions", "transferUVs"]
	pass

