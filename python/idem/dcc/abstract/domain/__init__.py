from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from ..session import DCCIdemSession

__dcc_session__ = None
def getDCCSession()->DCCIdemSession:
	return __dcc_session__