

from __future__ import annotations
import typing as T

from pathlib import Path
from PySide2 import QtCore, QtWidgets, QtGui
from wplib import log
from wpdex import *
from wpdex import react

from wp import constant
from wp.pipe.ui import AssetSelectorWidget
from wpdex.ui.atomic import *
from chimaera.ui import ChimaeraWidget
from wpui.widget import ScriptWidget, LogWidget, FileBrowserButton

from wpui import lib as uilib

from idem.model import IdemSession, IdemGraph

"""overall window representing single instance of Idem - 


TODO: 
    come back and rewrite this properly with fields and generated widgets
    
    
we might create the path widget thus - 

path = PathWidget(parent=None,
	default="strategy.chi",
	parentDir=lambda : ref("asset").diskPath(),
	postProcessFn=lambda rawPath : Path(rawPath).relativeTo(self.parentDir())
    
"""

class IdemWidget(QtWidgets.QWidget):
	"""on left, panel displying current target idem file
	on right, big graph
	console log on bottom?
	SCRIPTING window on bottom left?

	aero is fleeting
	mondrian is forever

	TODO:
		- serialisation -
			- consider the errors that can occur on loading file - assets might be invalid,
				python types might have moved in code -
				have UI to check for all of these, present cases to user, to be resolved
	"""

	def __init__(self,
	             session:IdemSession,
	             parent=None
	             ):
		QtWidgets.QWidget.__init__(self, parent=parent)
		self.session = session

		self.assetW = AssetSelectorWidget(
			value=self.session.data.ref("asset", "@V"),
			parent=self,
			name="asset",
		                                  )

		self.filePathW = FileStringWidget(
			value=self.session.data.ref("filePath", "@V"),

			parent=self,
			name="filePath",
			parentDir=self.session.data.ref("asset", "@V").rx.where(
				self.session.data.ref("asset", "@V").diskPath(),
				constant.ASSET_ROOT
			),
			dialogCaption="Select .chi file in asset",
			showRelativeFromParent=True,
			allowMultiple=False,
			)

		self.dccPalette = QtWidgets.QLabel("live DCCs go here", parent=self)

		self.scriptW = ScriptWidget(parent=self)

		self.logW = LogWidget(parent=self)

		# don't pass live reference, let graph widget do its own specialisation
		self.graphW = ChimaeraWidget(#session.data["graph"],
		                             value=session.ref("graph", "@V"),
		                             parent=self)

		#region layout
		l = QtWidgets.QGridLayout(self)
		l.addWidget(self.assetW, 0, 0, 1, 1)
		l.addWidget(self.filePathW, 1, 0, 1, 1)
		l.addWidget(self.dccPalette, 2, 0, 2, 1)
		l.addWidget(self.scriptW, 4, 0, 3, 1)

		l.addWidget(self.graphW, 0, 1, 4, 3)
		l.addWidget(self.logW, 4, 1, 1, 3)

		saveAction = QtWidgets.QAction(self)
		# QtCore.Qt.CTRL != QtCore.Qt.Key_Control - the more you know
		saveAction.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL +
		                                          QtCore.Qt.Key_S))
		saveAction.triggered.connect(self.saveSession)
		self.addAction(saveAction)

		openAction = QtWidgets.QAction(self)
		# QtCore.Qt.CTRL != QtCore.Qt.Key_Control - the more you know
		openAction.setShortcut(QtGui.QKeySequence(QtCore.Qt.CTRL +
		                                          QtCore.Qt.Key_O))
		openAction.triggered.connect(self.selectSessionAndOpen)
		self.addAction(openAction)

	def saveSession(self, toPath:Path=None):
		"""serialises the current session to the given path, or to the currently
		selected asset/file if none given
		"""
		self.session.saveSession(toPath=toPath)


	def selectSessionAndOpen(self, dirPath=None):
		result = QtWidgets.QFileDialog.getOpenFileName(
			parent=self,
			caption="Open chimaera session",
			dir=str(dirPath or self.session.fullChiPath().parent),
			filter="CHI file (*.chi)"
		) or ""
		if not result:
			return
		result = result[0]
		self.session.loadSession(result)





