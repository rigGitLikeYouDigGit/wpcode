

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
from .. import retriever
RenderLight = retriever.getNodeCls("RenderLight")
assert RenderLight
if T.TYPE_CHECKING:
	from .. import RenderLight

# add node doc



# region plug type defs
class DecayRatePlug(Plug):
	node : NonAmbientLightShapeNode = None
	pass
class EmitDiffusePlug(Plug):
	node : NonAmbientLightShapeNode = None
	pass
class EmitSpecularPlug(Plug):
	node : NonAmbientLightShapeNode = None
	pass
# endregion


# define node class
class NonAmbientLightShapeNode(RenderLight):
	decayRate_ : DecayRatePlug = PlugDescriptor("decayRate")
	emitDiffuse_ : EmitDiffusePlug = PlugDescriptor("emitDiffuse")
	emitSpecular_ : EmitSpecularPlug = PlugDescriptor("emitSpecular")

	# node attributes

	typeName = "nonAmbientLightShapeNode"
	typeIdInt = 1312902466
	pass

