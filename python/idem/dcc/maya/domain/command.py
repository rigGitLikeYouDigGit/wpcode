from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from idem.dcc.abstract.command import *

from wpm import cmds, om

"""test for structure here - 
DCC domain commands shadow names of abstract command bases, 
overriding execute() in domain-specific code - 
a DCC-side server will load the domain cmd just the same, 
and let us execute it

maybe also get to undo/redo

lozenge inheritance pattern here:
		 |  ( this side is maya domain )
DCCCmd
 |       \
 |          MayaCmd
 V              |
SetCameraCmd    |
		\       V
		(Maya)SetCameraCmd
		
"""

if T.TYPE_CHECKING:
	from .session import MayaIdemSession

class MayaCmd(DCCCmd):
	session : MayaIdemSession = None


class SetCameraCmd(SetCameraCmd, MayaCmd):
	def execute(self):
		pass

