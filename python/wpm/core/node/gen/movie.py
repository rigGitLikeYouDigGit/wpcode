

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core.node.base import om, WN, Plug, PlugDescriptor

# add any extra imports
if T.TYPE_CHECKING:
	from ..author import Catalogue
	File = Catalogue.File
else:
	from .. import retriever
	File = retriever.getNodeCls("File")
	assert File

# add node doc



# region plug type defs

# endregion


# define node class
class Movie(File):

	# node attributes

	typeName = "movie"
	typeIdInt = 1381256534
	nodeLeafClassAttrs = []
	nodeLeafPlugs = []
	pass

