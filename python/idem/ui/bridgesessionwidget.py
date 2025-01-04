from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets

from wplib import sequence
from wpui.widget import GenGroupBox
from idem.ui.sessionwidget import SessionWidget, ConnectedSessWidget, _SessionCtlWidget
from idem.dcc.abstract.session import IdemBridgeSession, DCCIdemSession, DataFileServer, getActivePortDataPathMap

class IdemBridgeWidget(SessionWidget):
	"""TEMP TEMP TEMP TEMP

	"""
	session : IdemBridgeSession
	def __init__(self, name="idemBRIDGE", parent=None):
		super().__init__(parent)
		# all we have to do is update the generation logic for the group boxes
		self.layout().addWidget(QtWidgets.QLabel("_bridge_", self))

		def _updateConnectedWidgetsFn(groupBox):
			"""show any sessions connected to this bridge"""
			result = []
			if not self.session:
				return []
			#log("update connected b",)
			#log("linked sessions:", self.session.linkedSessions)
			#log("availDCCSessions", DataFileServer.availableDCCSessions())
			for k, v in self.session.linkedSessions.items():
				w = ConnectedSessWidget(
					parent=groupBox,
					name=DataFileServer.availableDCCSessions()[k].stem,
					isConnected=True,
				)
				w.button.clicked.connect(
					lambda sender: self.onConnectBtnPressed(sender, w))
				result.append(w)
			#log("connected done")
			return result

		def _updateAvailWidgetsFn(groupBox):
			result = []

			for k, path in IdemBridgeSession.availableDCCSessions().items():
				if self.session:
					if k in self.session.linkedSessions:
						continue
				w = ConnectedSessWidget(
					parent=groupBox,
					name=DataFileServer.availableDCCSessions()[k].stem,
					isConnected=False,
				)
				w.button.clicked.connect(
					lambda sender: self.onConnectBtnPressed(sender, w))
				if not self.session:
					w.setEnabled(False)
				result.append(w)
			return result

		self.connectedGroupBox.getWidgetsFn = _updateConnectedWidgetsFn
		self.availGroupBox.getWidgetsFn = _updateAvailWidgetsFn

	def onConnectBtnPressed(self, sender, sessW:ConnectedSessWidget):
		"""connect target session to this bridge"""
		self.session.connectToSocket(sessW.port)
		self.sync()


	instance : IdemBridgeWidget = None

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
				"Idem",
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
			text = "bridge" + text # all bridge sessions start with it in name
			session = IdemBridgeSession.bootstrap(text)
			self.setSession(session)
			#self.setSession(sessionCls.bootstrap(text))



if __name__ == '__main__':

	app = QtWidgets.QApplication()
	w = IdemBridgeWidget.getInstance()
	w.show()
	w.setMinimumSize(200, 200)


	app.exec_()


