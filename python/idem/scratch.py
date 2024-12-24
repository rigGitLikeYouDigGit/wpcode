from __future__ import annotations

import socketserver
import time
import types, typing as T
import pprint
from wplib import log
import sys, os

from socketserver import TCPServer, ThreadingTCPServer


from wplib.network import libsocket
from idem.dcc.abstract.domain import DCCIdemSession
from idem.dcc.abstract.domain import session

#s = TCPServer(("", 0), socketserver.BaseRequestHandler)
# s = ThreadingTCPServer(("", 0), socketserver.BaseRequestHandler)
# s.serve_forever()
#
# raise

sessionA = DCCIdemSession(name="A")
log("before run A")
sessionA.runThreaded()


#sessionB = DCCIdemSession(sessionTempName="B")
#log("before run B")
#sessionB.runThreaded()
log("before send")
sessionA.send(sessionA.message({"my" : "message"}))
log("after send")

#print(libsocket.getOpenSockets())
#print(session.getActivePortDataPathMap())
session.clearInactiveDataFiles()

bridge = session.IdemBridgeSession()

bridge.connectToSocket(sessionA.portId())

