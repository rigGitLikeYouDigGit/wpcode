from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from idem.maya.dcc import Maya
from idem.node import DCCSessionNode

class MayaSessionNode(DCCSessionNode):
	@classmethod
	def dccType(cls)->type[Maya]:
		return Maya
