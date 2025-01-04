from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

import time
from idem.dcc.abstract import DCCIdemSession

s : DCCIdemSession = DCCIdemSession.bootstrap("testSession", start=True)

time.sleep(3)
log("avail", s.availableBridgeSessions())

log("before DCC connect")
s.updateConnectedBridge(
	tuple(s.availableBridgeSessions().keys())[0],
	sendCmd=True
)

log("after connect")
# while True:
#	time.sleep(1)

