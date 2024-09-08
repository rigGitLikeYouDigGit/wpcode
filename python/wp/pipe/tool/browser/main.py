from __future__ import annotations

from PySide2 import QtWidgets

from wpui import layout
from wp.ui.widget import ChildListWidget

from wpasset.bank import AssetBank
from wpasset import Asset
from wpasset.ui.widget import UidWidget

"""show what assets are available in the project"""

class DictTableWidget(QtWidgets.QTableWidget):
	def __init__(self, parent=None):
		super(DictTableWidget, self).__init__(parent)

		self.setColumnCount(2)
		self.horizontalHeader().setStretchLastSection(True)
		self.verticalHeader().setVisible(False)
		self.horizontalHeader().setVisible(False)

	def setDict(self, d:dict):
		self.clear()
		self.setRowCount(len(d))
		for i, (k, v) in enumerate(d.items()):
			self.setItem(i, 0, QtWidgets.QTableWidgetItem(k))
			self.setItem(i, 1, QtWidgets.QTableWidgetItem(str(v)))
		self.resizeColumnsToContents()
		self.resizeRowsToContents()


	def getResultDict(self)->dict:
		"""get the dict from the table"""
		d = {}
		for i in range(self.rowCount()):
			d[self.item(i, 0).text()] = self.item(i, 1).text()
		return d


class AssetEntryWidget(QtWidgets.QWidget):
	"""display data on single asset entry
	pretty much just tags, nice name and uid"""

	def __init__(self, parent=None):
		super().__init__(parent=parent)
		self.nameLabel = QtWidgets.QLabel(parent=self)
		self.uidLabel = UidWidget(parent=self)
		self.tagsWidget = DictTableWidget(parent=self)
		self._asset : Asset = None
		self.makeLayout()

	def makeLayout(self):
		hl = QtWidgets.QHBoxLayout()
		hl.addWidget(self.nameLabel)
		hl.addWidget(self.uidLabel)
		hl.addWidget(self.tagsWidget)
		self.setLayout(hl)

	def setAsset(self, asset:Asset):
		self._asset = asset
		#self.nameLabel.setText(self._asset.name)
		self.uidLabel.setUid(self._asset.uid)
		self.tagsWidget.setDict(self._asset.tags)
		#self.makeLayout()



class AssetBrowserWidget(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent=parent)

		self._bank = AssetBank()

		self.refreshBtn = QtWidgets.QPushButton("Refresh", parent=self)
		self.assetList = ChildListWidget(parent=self)

		self.makeLayout()
		self.makeConnections()

	def bank(self)->AssetBank:
		return self._bank

	def refresh(self):
		"""update with list of existing assets"""
		self.assetList.clear()
		self._bank = AssetBank()
		self.bank().refreshFromAssetDir()
		print(self.bank().internalDataDir(),		      )
		print(self.bank().internalSearchDataDir())
		print(self.bank().assets)
		for uid, asset in self.bank().assets.items():
			assetWidget = AssetEntryWidget(parent=self.assetList)
			#assetWidget.nameLabel.setText(asset)
			self.assetList.addWidget(assetWidget)

			assetWidget.setAsset(asset)

	def _onRefreshPressed(self, *args, **kwargs):
		self.refresh()


	def makeLayout(self):
		layout.genAutoLayout(self)

	def makeConnections(self):
		self.refreshBtn.clicked.connect(self._onRefreshPressed)


	pass


if __name__ == '__main__':
	import sys
	from PySide2.QtWidgets import QApplication

	app = QApplication(sys.argv)
	widget = AssetBrowserWidget()
	widget.show()
	sys.exit(app.exec_())

