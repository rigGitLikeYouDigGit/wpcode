

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	Facade = Catalogue.Facade
else:
	from .. import retriever
	Facade = retriever.getNodeCls("Facade")
	assert Facade

# add node doc



# region plug type defs

# endregion


# define node class
class EnvFacade(Facade):

	# node attributes

	typeName = "envFacade"
	apiTypeInt = 976
	apiTypeStr = "kEnvFacade"
	typeIdInt = 1380271683
	MFnCls = om.MFnDependencyNode
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

