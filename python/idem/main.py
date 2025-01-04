from __future__ import annotations
import types, typing as T
import pprint

import idem.dcc.abstract.bridge
from wplib import log


"""temp file to run an idem bridge in headless mode

"""#


from wplib.network import libsocket
from idem.dcc.abstract import DCCIdemSession
from idem.dcc.abstract import session

#s = TCPServer(("", 0), socketserver.BaseRequestHandler)
# s = ThreadingTCPServer(("", 0), socketserver.BaseRequestHandler)
# s.serve_forever()
#

#session.clearInactiveDataFiles()

bridge = idem.dcc.abstract.bridge.IdemBridgeSession()
bridge.runThreaded()


