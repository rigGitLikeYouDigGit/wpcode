

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Reflect = Catalogue.Reflect
else:
	from .. import retriever
	Reflect = retriever.getNodeCls("Reflect")
	assert Reflect

# add node doc



# region plug type defs
class EccentricityPlug(Plug):
	node : Blinn = None
	pass
class ReflectionRolloffPlug(Plug):
	node : Blinn = None
	pass
class SpecularRollOffPlug(Plug):
	node : Blinn = None
	pass
# endregion


# define node class
class Blinn(Reflect):
	eccentricity_ : EccentricityPlug = PlugDescriptor("eccentricity")
	reflectionRolloff_ : ReflectionRolloffPlug = PlugDescriptor("reflectionRolloff")
	specularRollOff_ : SpecularRollOffPlug = PlugDescriptor("specularRollOff")

	# node attributes

	typeName = "blinn"
	apiTypeInt = 373
	apiTypeStr = "kBlinn"
	typeIdInt = 1380076622
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = ["eccentricity", "reflectionRolloff", "specularRollOff"]
	nodeLeafPlugs = ["eccentricity", "reflectionRolloff", "specularRollOff"]
	pass

