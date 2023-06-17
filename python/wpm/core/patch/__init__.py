
"""explicit package holding patched / wrapped versions of normal
Maya modules - cmds, om, etc

all WP Maya code should import these (from core, or wpm)
"""

from . import hosttarget
from . import wrap

wrap.wrapCmds(hosttarget, "baseCmds", "wpCmds")
wrap.wrapOm(hosttarget, "baseOm", "wpOm")

#import newly wrapped modules with new names
from .hosttarget import (
	wpCmds as cmds,
	wpOm as om,
	wpOmr as omr,
	wpOma as oma,
	wpOmui as omui
)
