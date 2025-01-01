from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from .abstract import *




def currentDCCProcessCls()->type[DCCProcess]:
	"""return the class of the DCC process
	fitting the current working environment"""
	for i in DCCProcess.__subclasses__():
		if i.isThisCurrentDCC():
			return i
	return DCCProcess

from .houdini import *
from .maya import *
