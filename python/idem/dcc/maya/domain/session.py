from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from idem.dcc.abstract import DCCIdemSession

from wpm import cmds, om, WN

class MayaIdemSession(DCCIdemSession):
	""""""

	def getSessionCamera(self)->WN:
		"""for now, just return the persp camera -
		should probably make an "idem_CAM" node
		later on
		"""

