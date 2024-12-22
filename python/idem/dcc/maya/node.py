from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from idem.dcc.maya import MayaProcess
from idem.node import DCCSessionNode

class MayaSessionNode(DCCSessionNode):
	@classmethod
	def dccProcessType(cls)->type[Maya]:
		return Maya
