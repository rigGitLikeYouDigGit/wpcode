from __future__ import annotations

import socketserver
import time
import types, typing as T
import pprint
from wplib import log
import sys, os

from socketserver import TCPServer, ThreadingTCPServer


from wplib.network import libsocket
from idem.dcc.abstract import DCCIdemSession
from idem.dcc.abstract import session


import asyncio


"""test for how async works with blocking
we want behaviour identical to a thread, where an outer,
synchronous script can set up an async loop running when main processes are idle 
"""

async def loopFn():

	while True:
		print("looping async")
		await asyncio.sleep(1.5)



def mainScript():

	print("before any async")
	#asyncio.run(loopFn())
	# loopFn()

	loop = asyncio.get_event_loop()
	loop.run_until_complete(loopFn())


	#time.sleep(5)
	print("after any sync")
	print("script ends")



mainScript()


#s = TCPServer(("", 0), socketserver.BaseRequestHandler)
# s = ThreadingTCPServer(("", 0), socketserver.BaseRequestHandler)
# s.serve_forever()
#

# session.clearInactiveDataFiles()
#
# sessionA = DCCIdemSession(name="A")
# log("before run A")
# sessionA.runThreaded()
#
#
# #sessionB = DCCIdemSession(sessionTempName="B")
# #log("before run B")
# #sessionB.runThreaded()
# log("before send")
# sessionA.send(sessionA.message({"my" : "message"}))
# log("after send")
#
# #print(libsocket.getOpenSockets())
# #print(session.getActivePortDataPathMap())
#
# bridge = session.IdemBridgeSession()
# bridge.runThreaded()
#
# bridge.connectToSocket(sessionA.portId())

