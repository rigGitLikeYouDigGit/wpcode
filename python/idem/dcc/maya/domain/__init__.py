from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

"""code inside this package may import dcc-specific 
packages - this will never be called in the pure python bridge"""

from .session import MayaIdemSession

__dcc_session__ = None
def getDCCSession()->MayaIdemSession:
	return __dcc_session__