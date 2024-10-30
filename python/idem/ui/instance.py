

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
from wpui.widget import ScriptWidget, LogWidget

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
	"""

	def __init__(self,
	             session:IdemSession,
	             parent=None
	             ):
		QtWidgets.QWidget.__init__(self, parent=parent)
		self.session = session


		ref = self.session.data.ref("asset", "@V")

		# log("check root", ref,
		#     react.canBeSet(ref)
		#     )
		# #ref.rx.value = "asdasd"
		# assert react.canBeSet(ref)


		self.assetW = AssetSelectorWidget(
			value=self.session.data.ref("asset", "@V"),
			parent=self,
			name="asset",
		                                  )
		#return
		#assetRef = self.session.data.ref("asset", "@V")
		#log("asset ref", assetRef.rx.value)
		#pathRef = self.session.data.ref("filePath", "@V")
		#log("file path", pathRef.rx.value, react.canBeSet(pathRef))
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

		self.dccPalette = QtWidgets.QLabel("liveDccs go here", parent=self)

		self.scriptW = ScriptWidget(parent=self)

		self.logW = LogWidget(parent=self)

		# don't pass live reference, let graph widget do its own specialisation
		self.graphW = ChimaeraWidget(session.data["graph"], parent=self)

		#region layout
		l = QtWidgets.QGridLayout(self)
		l.addWidget(self.assetW, 0, 0, 1, 1)
		l.addWidget(self.filePathW, 1, 0, 1, 1)
		l.addWidget(self.dccPalette, 2, 0, 2, 1)
		l.addWidget(self.scriptW, 4, 0, 3, 1)

		l.addWidget(self.graphW, 0, 1, 4, 3)
		l.addWidget(self.logW, 4, 1, 1, 3)

		# test
		#log("get root")
		#log(uilib.rootWidget(self.filePathW))
		#log(uilib.widgetParents(self.filePathW))





