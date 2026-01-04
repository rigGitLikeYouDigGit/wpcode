

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	_BASE_ = Catalogue._BASE_
else:
	from .. import retriever
	_BASE_ = retriever.getNodeCls("_BASE_")
	assert _BASE_

# add node doc



# region plug type defs
class BinMembershipPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class CmEnabledPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class ConfigFileEnabledPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class ConfigFilePathPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class DefaultInputSpaceNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class DisplayNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class OutputTransformEnabledPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class OutputTransformNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class OutputTransformUseColorConversionPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class OutputUseViewTransformPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class PlayblastOutputTransformEnabledPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class PlayblastOutputTransformNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class PlayblastOutputTransformUseColorConversionPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class PlayblastOutputUseViewTransformPlug(Plug):
	node : ColorManagementGlobals = None
	pass
class ViewNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class ViewTransformNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
class WorkingSpaceNamePlug(Plug):
	node : ColorManagementGlobals = None
	pass
# endregion


# define node class
class ColorManagementGlobals(_BASE_):
	binMembership_ : BinMembershipPlug = PlugDescriptor("binMembership")
	cmEnabled_ : CmEnabledPlug = PlugDescriptor("cmEnabled")
	configFileEnabled_ : ConfigFileEnabledPlug = PlugDescriptor("configFileEnabled")
	configFilePath_ : ConfigFilePathPlug = PlugDescriptor("configFilePath")
	defaultInputSpaceName_ : DefaultInputSpaceNamePlug = PlugDescriptor("defaultInputSpaceName")
	displayName_ : DisplayNamePlug = PlugDescriptor("displayName")
	outputTransformEnabled_ : OutputTransformEnabledPlug = PlugDescriptor("outputTransformEnabled")
	outputTransformName_ : OutputTransformNamePlug = PlugDescriptor("outputTransformName")
	outputTransformUseColorConversion_ : OutputTransformUseColorConversionPlug = PlugDescriptor("outputTransformUseColorConversion")
	outputUseViewTransform_ : OutputUseViewTransformPlug = PlugDescriptor("outputUseViewTransform")
	playblastOutputTransformEnabled_ : PlayblastOutputTransformEnabledPlug = PlugDescriptor("playblastOutputTransformEnabled")
	playblastOutputTransformName_ : PlayblastOutputTransformNamePlug = PlugDescriptor("playblastOutputTransformName")
	playblastOutputTransformUseColorConversion_ : PlayblastOutputTransformUseColorConversionPlug = PlugDescriptor("playblastOutputTransformUseColorConversion")
	playblastOutputUseViewTransform_ : PlayblastOutputUseViewTransformPlug = PlugDescriptor("playblastOutputUseViewTransform")
	viewName_ : ViewNamePlug = PlugDescriptor("viewName")
	viewTransformName_ : ViewTransformNamePlug = PlugDescriptor("viewTransformName")
	workingSpaceName_ : WorkingSpaceNamePlug = PlugDescriptor("workingSpaceName")

	# node attributes

	typeName = "colorManagementGlobals"
	typeIdInt = 1129137986
	nodeLeafClassAttrs = ["binMembership", "cmEnabled", "configFileEnabled", "configFilePath", "defaultInputSpaceName", "displayName", "outputTransformEnabled", "outputTransformName", "outputTransformUseColorConversion", "outputUseViewTransform", "playblastOutputTransformEnabled", "playblastOutputTransformName", "playblastOutputTransformUseColorConversion", "playblastOutputUseViewTransform", "viewName", "viewTransformName", "workingSpaceName"]
	nodeLeafPlugs = ["binMembership", "cmEnabled", "configFileEnabled", "configFilePath", "defaultInputSpaceName", "displayName", "outputTransformEnabled", "outputTransformName", "outputTransformUseColorConversion", "outputUseViewTransform", "playblastOutputTransformEnabled", "playblastOutputTransformName", "playblastOutputTransformUseColorConversion", "playblastOutputUseViewTransform", "viewName", "viewTransformName", "workingSpaceName"]
	pass

