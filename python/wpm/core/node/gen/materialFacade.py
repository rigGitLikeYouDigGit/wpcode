

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
Facade = retriever.getNodeCls("Facade")
assert Facade
if T.TYPE_CHECKING:
	from .. import Facade

# add node doc



# region plug type defs
class HardwareProxyPlug(Plug):
	node : MaterialFacade = None
	pass
class OutColorBPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : MaterialFacade = None
	pass
class OutColorGPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : MaterialFacade = None
	pass
class OutColorRPlug(Plug):
	parent : OutColorPlug = PlugDescriptor("outColor")
	node : MaterialFacade = None
	pass
class OutColorPlug(Plug):
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	ocb_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	ocg_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	ocr_ : OutColorRPlug = PlugDescriptor("outColorR")
	node : MaterialFacade = None
	pass
class ProxyInitProcPlug(Plug):
	node : MaterialFacade = None
	pass
# endregion


# define node class
class MaterialFacade(Facade):
	hardwareProxy_ : HardwareProxyPlug = PlugDescriptor("hardwareProxy")
	outColorB_ : OutColorBPlug = PlugDescriptor("outColorB")
	outColorG_ : OutColorGPlug = PlugDescriptor("outColorG")
	outColorR_ : OutColorRPlug = PlugDescriptor("outColorR")
	outColor_ : OutColorPlug = PlugDescriptor("outColor")
	proxyInitProc_ : ProxyInitProcPlug = PlugDescriptor("proxyInitProc")

	# node attributes

	typeName = "materialFacade"
	apiTypeInt = 975
	apiTypeStr = "kMaterialFacade"
	typeIdInt = 1380795971
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["hardwareProxy", "outColorB", "outColorG", "outColorR", "outColor", "proxyInitProc"]
	nodeLeafPlugs = ["hardwareProxy", "outColor", "proxyInitProc"]
	pass

