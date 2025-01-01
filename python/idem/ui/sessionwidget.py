from __future__ import annotations

import time
import types, typing as T
import pprint
from wplib import log

import asyncio # I've never typed this import before :D
""" after some education, asyncio seems the wrong solution to spawn independent
workers WITHOUT blocking the main thread - we don't want to suddenly turn the
whole of Houdini or Maya into a python async application, so the main thread
(that also runs the script editor) has to remain open after running this setup,
and sending various bits of idem loose into the program.

to spawn this separate monitor process, use a thread.

it MIGHT be an idea to reduce thread usage for idem to have it structured as
async, with each instance running off a maximum of a single thread.
might be a good idea actually.


main thread
|
V
startIdem():
	make a thread
|					\ 
|                     IDEM():  
V                         set up async event loop
continue on,                do loads of async / await stuff
do whatever,                  do whatever
live your life,   


we also need a layer of separation between the callbacks in the DCC and 
the direct network stuff, Houdini especially seems very fragile to
callbacks like that


"""

import threading
from PySide2 import QtCore, QtGui, QtWidgets

from idem.dcc.abstract import DCCIdemSession, IdemBridgeSession, DataFileServer
from idem.dcc.abstract import session

from idem.ui.sessionstatus import BlinkLight

class ConnectedSessWidget(QtWidgets.QWidget):
	"""Show label representing remote session alongside
	button to affect connection - if available, show 'connect'
	if already connected, show 'disconnect'

	no deep object linked, just fire signals back to parent widget
	"""

	def __init__(self, parent=None,
	             name="", isConnected=False,
	             #onClickFn:T.Callable=None
	             ):
		super().__init__(parent)
		vl = QtWidgets.QHBoxLayout()
		self.isConnected = isConnected
		self.name = name
		self.label = QtWidgets.QLabel(name, self)
		self.button = QtWidgets.QPushButton("disconnect" if isConnected else "connect")
		vl.addWidget(self.label)
		vl.addWidget(self.button)
		self.setLayout(vl)
		#self.button.clicked.connect(onClickFn)

class _SessionCtlWidget(QtWidgets.QWidget):
	"""widget specifically to control the current dcc session"""
	def __init__(self, parent=None):
		super().__init__(parent)
		self.btn = QtWidgets.QPushButton("start server", self)
		vl = QtWidgets.QVBoxLayout(self)
		vl.addWidget(self.btn)
		self.setLayout(vl)

	def setHasSession(self, state):
		if state:
			self.btn.setText("stop server")
		else:
			self.btn.setText("start server")



class SessionWidget(QtWidgets.QGroupBox):
	"""show session name, if it's active
	show dropdown of connected sessions

	I guess the fully correct way would be to make a session Modelled, and
	have all this stuff update reactively,
	but come on
	"""

	instance = None

	def __init__(self, parent=None):
		super().__init__(parent)
		self.setTitle("IDEM")
		self.session : DataFileServer = None
		self.nameLabel = QtWidgets.QLabel("name", self)

		self.indicator = BlinkLight(parent=self,
		                            value=BlinkLight.Status.Success,
		                            size=6,
		                            decayLength=0.7
		                            )
		self.indicator.move(1, 4)

		self.sessionCtlW = _SessionCtlWidget(parent=self)

		self.connectedGroupBox = QtWidgets.QGroupBox("connected:", parent=self)
		self.availGroupBox = QtWidgets.QGroupBox("available:", parent=self)

		self.shouldStop = False
		self.syncThread :threading.Thread = None

		self.connectedVL = QtWidgets.QVBoxLayout(self.connectedGroupBox)
		self.availVL = QtWidgets.QVBoxLayout(self.availGroupBox)

		self.connectedGroupBox.setLayout(self.connectedVL)
		self.availGroupBox.setLayout(self.availVL)

		vl = QtWidgets.QVBoxLayout()
		vl.addWidget(self.nameLabel)
		vl.addWidget(self.sessionCtlW)
		vl.addWidget(self.connectedGroupBox)
		vl.addWidget(self.availGroupBox)
		self.setLayout(vl)

	def setSession(self, session:DataFileServer):
		"""update the session for this widget to focus on"""
		self.session = session
		self.sync()

	def _onSessionCtlWBtnPress(self, *args, **kwargs):
		if self.session: # end current session
			self.session.clear()
			self.sessionCtlW.setHasSession(False)
			self.sync()
		else: # start new session



	def sync(self):
		#print("sync")
		#self.indicator.setValue(BlinkLight.Status.Success)
		self.indicator.blink()
		#print("after set value")
		#return
		if self.session:
			self.nameLabel.setText(str(self.session._sessionFileName()))

		else:
			self.nameLabel.setText("<no session>")
		self.syncAvailWidgets()
		#self.indicator.blink()
		self.indicator.setValue(BlinkLight.Status.Success)
		#print("blinked")

	def stop(self):
		self.shouldStop = True
		#asyncio

	# async def syncForever(self):
	# 	"""first ever time using async, sure this will be fine
	# 	"""

	def syncThreaded(self):
		while not self.shouldStop:
			self.sync()
			time.sleep(1.0)

	def syncForever(self):
		self.syncThread = threading.Thread(target=self.syncThreaded,
		                                   daemon=True)
		self.syncThread.start()
		#self.syncThreaded()

	def connectedWidgets(self):
		return [self.connectedVL.itemAt(i) for i in range(self.connectedVL.count())]
	def availWidgets(self):
		return [self.availVL.itemAt(i) for i in range(self.availVL.count())]

	def onConnectBtnPressed(self, sender, sessW:ConnectedSessWidget):
		"""check if widget was a linked one or not"""
		if sessW.isConnected: # disconnect
			if sessW in self.connectedWidgets():
				if isinstance(self.session, DCCIdemSession):
					self.session.disconnectBridge(sendCmd=True)
			self.sync()
			return
		if sessW in self.availWidgets(): # connect
			if isinstance(self.session, DCCIdemSession):
				self.session.updateConnectedBridge(int(sessW.name), sendCmd=True)
			self.sync()

	def syncAvailWidgets(self):
		"""TODO: would be nice to have some kind of generative layout object,
					 not the first time we've had this updating list behaviour
					 """
		for i in self.connectedWidgets():
			self.connectedVL.removeWidget(i)
		for i in self.availWidgets():
			self.availVL.removeWidget(i)

		if isinstance(self.session, DCCIdemSession):
			if self.session.connectedBridgeId():
				w = ConnectedSessWidget(
					parent=self.connectedGroupBox,
					name=str(self.session.connectedBridgeId()),
					isConnected=True,
				)
				w.button.clicked.connected(
					lambda sender: self.onConnectBtnPressed(sender, w))
				self.connectedVL.addWidget(w)
			for k, path in self.session.availableBridgeSessions().items():
				w = ConnectedSessWidget(
					parent=self.availGroupBox,
					name=str(k),
					isConnected=False,
				)
				w.button.clicked.connected(
					lambda sender: self.onConnectBtnPressed(sender, w))
				self.availVL.addWidget(w)
			return

		# if no session set up yet, show available processes
		if self.session is None:
			activeMap = session.getActivePortDataPathMap()
			for portId, path in activeMap.items():
				w = ConnectedSessWidget(
					parent=self.availGroupBox,
					name=str(portId),
					isConnected=False,
				)
				w.button.clicked.connected(
					lambda sender: self.onConnectBtnPressed(sender, w))
				self.availVL.addWidget(w)
			return


	@classmethod
	def getInstance(cls)->SessionWidget:
		if cls.instance: return cls.instance
		cls.instance = SessionWidget()
		return cls.instance


if __name__ == '__main__':

	app = QtWidgets.QApplication()
	w = SessionWidget.getInstance()
	w.show()
	w.syncForever()
	app.exec_()



