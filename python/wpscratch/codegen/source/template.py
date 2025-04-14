

from __future__ import annotations
import typing as T

import numpy as np

from wpm.core import cmds, om, WN, WPlug
#from wpm.core.node.base import WN, PlugDescriptor, Plug
from wpm.core.node.base import *

# add any extra imports
{IMPORT_BLOCK}

# add node doc
{DOC_BLOCK}


# region plug type defs
{ATTR_DEF_BLOCK}
# endregion


# define node class
{NODE_DEF_BLOCK}

