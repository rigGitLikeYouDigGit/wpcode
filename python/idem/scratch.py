from __future__ import annotations

import socketserver
import time
import types, typing as T
import pprint
from wplib import log

from socketserver import TCPServer, ThreadingTCPServer

from idem.dcc.abstract.domain import DCCIdemSession

#s = TCPServer(("", 0), socketserver.BaseRequestHandler)
# s = ThreadingTCPServer(("", 0), socketserver.BaseRequestHandler)
# s.serve_forever()
#
# raise

sessionA = DCCIdemSession(sessionTempName="A")

#sessionA.server.serve_forever(0.1)

#sessionA.server.server_close()

sessionB = DCCIdemSession(sessionTempName="B")

log("before run A")
sessionA.runThreaded()
log("before run B")
sessionB.runThreaded()
log("before send")
sessionA.send("hello")
log("after send")
#time.sleep(10)
