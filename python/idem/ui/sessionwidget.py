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

from wplib import sequence
from wpui.widget import GenGroupBox
from idem.dcc.abstract import DCCIdemSession, DataFileServer
from idem.dcc.abstract import session
from idem.dcc import DCCProcess, IdemBridgeSession

from idem.ui.sessionstatus import BlinkLight

class ConnectedSessWidget(QtWidgets.QWidget):
	"""Show label representing remote session alongside
	button to affect connection - if available, show 'connect'
	if already connected, show 'disconnect'

	no deep object linked, just fire signals back to parent widget
	"""

	def __init__(self, parent=None,
	             name="", isConnected=False,
	             port:int=-1
	             #onClickFn:T.Callable=None
	             ):
		super().__init__(parent)
		vl = QtWidgets.QHBoxLayout()
		self.isConnected = isConnected
		self.name = name
		self.port = port
		self.label = QtWidgets.QLabel(name, self)
		self.button = QtWidgets.QPushButton("disconnect" if isConnected else "connect")
		self.button.setContentsMargins(1, 1, 1, 1)
		vl.addWidget(self.label)
		vl.addWidget(self.button)
		self.label.setIndent(1)
		self.label.setFixedSize(self.label.sizeHint())
		textSize = self.button.fontMetrics().size(QtCore.Qt.TextShowMnemonic, self.button.text())
		self.button.setMinimumSize(textSize)
		self.label.setSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding,
		                         QtWidgets.QSizePolicy.Fixed,
		                         )
		self.button.setSizePolicy(QtWidgets.QSizePolicy.Maximum,
		                          QtWidgets.QSizePolicy.Fixed,
		                          )
		self.setLayout(vl)
		vl.setContentsMargins(0, 0, 0, 0)

		#self.setContentsMargins(5, 5, 5, 5)



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

	I don't know how, but I forgot you can't create QT widgets from
	a separate thread to the UI -
	maybe we can shunt the entire thing to a thread?

	but it feels like we're heading for async stuff sooner than planned
	"""

	instance = None

	def __init__(self, parent=None, name="IDEM"):
		super().__init__(parent)
		self.setTitle(name)
		self.session : DataFileServer = DataFileServer.session()
		self.nameLabel = QtWidgets.QLabel("name", self)

		self.delay = 1.0

		self.indicator = BlinkLight(parent=self,
		                            value=BlinkLight.Status.Success,
		                            size=6,
		                            decayLength=0.7
		                            )
		self.indicator.move(1, 4)

		self.sessionCtlW = _SessionCtlWidget(parent=self)
		self.sessionCtlW.btn.clicked.connect(self._onSessionCtlWBtnPress)

		def _updateConnectedWidgetsFn(groupBox):
			result = []
			if not self.session:
				return []
			if self.session.connectedBridgeId():
				w = ConnectedSessWidget(
					parent=groupBox,
					name=str(self.session.connectedBridgeId()),
					isConnected=True,
					port=self.session.connectedBridgeId()
				)
				#log("connected w", w)
				w.show()
				#log("w shown", w)
				w.button.clicked.connect(
					lambda sender: self.onConnectBtnPressed(sender, w))
				#self.connectedVL.addWidget(w)
				result.append(w)
			return result

		def _updateAvailWidgetsFn(groupBox):
			result = []
			# if not self.session:
			# 	return []
			print("availBridges", DCCIdemSession.availableBridgeSessions())

			for k, path in DCCIdemSession.availableBridgeSessions().items():
				w = ConnectedSessWidget(
					parent=groupBox,
					name=path.stem,
					isConnected=False,
					port=k
				)
				#log("avail w", w, w.isHidden())
				w.button.clicked.connect(
					lambda sender: self.onConnectBtnPressed(sender, w))

				#self.availVL.addWidget(w)
				if not self.session:
					w.setEnabled(False)
				result.append(w)
			#log("avail result", result)
			return result

		self.connectedGroupBox = GenGroupBox(
			"connected:", parent=self,
			getWidgetsFn=_updateConnectedWidgetsFn
		)
		self.availGroupBox = GenGroupBox(
			"available:", parent=self,
			getWidgetsFn=_updateAvailWidgetsFn
		)

		self.shouldStop = False
		self.syncThread :threading.Thread = None

		vl = QtWidgets.QVBoxLayout()
		vl.addWidget(self.nameLabel)
		vl.addWidget(self.sessionCtlW)
		vl.addWidget(self.connectedGroupBox)
		vl.addWidget(self.availGroupBox)
		self.setLayout(vl)

	def setSession(self, session:DataFileServer):
		"""update the session for this widget to focus on"""
		self.session = session
		self.sessionCtlW.setHasSession(True)
		self.sync()

	def _onSessionCtlWBtnPress(self, *args, **kwargs):
		if self.session: # end current session
			self.session.clear()
			self.sessionCtlW.setHasSession(False)
			self.session = None
			self.sync()
		else: # start new session
			# get user input for new session name
			# when launched from a bridge this will already have been set
			#TODO: I don't like the aesthetics of the "ok" "cancel" buttons qt gives by default
			text, ok = QtWidgets.QInputDialog.getText(None,
				"Enter IDEM session name",
				"",
				QtWidgets.QLineEdit.EchoMode.Normal,
				DCCProcess.currentDCCProcessCls().dccName + "Idem",
				QtCore.Qt.Popup,
				QtCore.Qt.ImhNone
			)
			if not ok: #cancelled, don't do anything
				log("cancelled, no session created")
				return
			if not text:
				log("Idem session must have a name")
				return
			# validate
			blockChars = " _/\\\'\"\n"
			if sequence.anyIn(blockChars, text):
				log("Invalid character entered, Idem session cannot contain: " + blockChars)
				return

			# bootstrap new session for this dcc
			sessionCls = DCCProcess.idemSessionCls()
			self.setSession(sessionCls.bootstrap(text))


	def sync(self, *args, **kwargs):
		#log("sync")
		#self.indicator.setValue(BlinkLight.Status.Success)
		self.indicator.blink()
		#print("after set value")
		#return
		if self.session:
			self.nameLabel.setText(str(self.session._sessionFileName()))

		else:
			self.nameLabel.setText("<no session>")
		self.syncAvailWidgets()
		#log("after sync avail")
		#self.indicator.blink()
		#self.indicator.setValue(BlinkLight.Status.Success)
		#print("blinked")

	def stop(self):
		self.shouldStop = True
		#asyncio

	# async def syncForever(self):
	# 	"""first ever time using async, sure this will be fine
	# 	"""

	def syncThreaded(self, *args, **kwargs):
		while not self.shouldStop:
			self.sync()
			time.sleep(self.delay)

	def syncForever(self, delay=1.0):
		# self.delay = delay
		# self.syncThread = threading.Thread(target=self.syncThreaded,
		#                                    daemon=True)
		# self.syncThread.start()

		#self.syncThreaded()

		self.timer = QtCore.QTimer(parent=self)
		self.timer.setInterval(1000)
		self.timer.timeout.connect(self.sync)

		self.timer.start()

	def connectedWidgets(self):
		#return [self.connectedVL.itemAt(i).widget() for i in range(self.connectedVL.count())]
		return self.connectedGroupBox.widgets()
	def availWidgets(self):
		return self.availGroupBox.widgets()

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
				self.session.updateConnectedBridge(int(sessW.port), sendCmd=True)
			self.sync()

	def syncAvailWidgets(self):
		"""					 """
		self.availGroupBox.sync()
		#log("after syncAvail gb")
		self.connectedGroupBox.sync()

	@classmethod
	def getInstance(cls)->SessionWidget:
		if cls.instance: return cls.instance
		cls.instance = cls()
		return cls.instance

	def closeEvent(self, event):
		"""pause the sync pings if you close the window"""
		self.shouldStop = True
		if self.syncThread is not None:
			self.syncThread.join()
		self.timer.stop()
		if self.session:
			self.session.clear()
		super().closeEvent(event)

	def showEvent(self, event):
		"""restart syncing"""
		self.shouldStop = False
		super().showEvent(event)
		self.syncForever()

	def __del__(self):
		self.shouldStop = True
		if self.syncThread is not None:
			self.syncThread.join()


w = None
def setupApp():
	app = QtWidgets.QApplication()
	global w
	w = SessionWidget.getInstance()
	w.show()
	w.setMinimumSize(200, 200)
	app.exec_()

if __name__ == '__main__':

	uiThread = threading.Thread(target=setupApp,
	                            daemon=False)
	uiThread.start()

	time.sleep(1)
	#w.show()
	#w.syncForever()


	# app = QtWidgets.QApplication()
	# w = SessionWidget.getInstance()
	# w.show()
	# w.setMinimumSize(200, 200)

	#app.exec_()

	# uiThread = threading.Thread(target=app.exec_,
	#                             daemon=True)
	# uiThread.start()

	# while True:
	# 	time.sleep(1)



