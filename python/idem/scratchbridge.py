from __future__ import annotations
import types, typing as T
import pprint
from wplib import log
import time
from idem.dcc import IdemBridgeSession

s : IdemBridgeSession = IdemBridgeSession.bootstrap("bridgeTest", start=True)

#time.sleep(3)
# log("avail", s.availableBridgeSessions())
#
# log("before DCC connect")
# s.updateConnectedBridge(tuple(s.availableBridgeSessions().keys())[0])
#
# log("after connect")
while True:
	time.sleep(1)