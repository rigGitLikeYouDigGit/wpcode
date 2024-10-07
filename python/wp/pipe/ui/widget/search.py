
from __future__ import annotations

import typing as T


from PySide2 import QtWidgets, QtCore, QtGui


class AssetSearchWidget(QtWidgets.QWidget):
	"""widget for searching assets"""

	def __init__(self, parent: T.Optional[QtWidgets.QWidget] = None):
		super(AssetSearchWidget, self).__init__(parent=parent)

		self.searchLineEdit = QtWidgets.QLineEdit()
		self.searchLineEdit.setPlaceholderText("Search")

		self.searchButton = QtWidgets.QPushButton("Search")
		self.searchButton.clicked.connect(self.onSearchButtonClicked)

		self.searchLayout = QtWidgets.QHBoxLayout()
		self.searchLayout.addWidget(self.searchLineEdit)
		self.searchLayout.addWidget(self.searchButton)

		self.searchResultsList = QtWidgets.QListWidget()
		self.searchResultsList.setSelectionMode(
			QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
		)
		self.searchResultsList.itemSelectionChanged.connect(
			self.onSearchResultsSelectionChanged
		)

		self.searchResultsLayout = QtWidgets.QVBoxLayout()
		self.searchResultsLayout.addWidget(self.searchResultsList)

		self.searchResultsWidget = QtWidgets.QWidget()
		self.searchResultsWidget.setLayout(self.searchResultsLayout)

		self.mainLayout = QtWidgets.QVBoxLayout()
		self.mainLayout.addLayout(self.searchLayout)
		self.mainLayout.addWidget(self.searchResultsWidget)

		self.setLayout(self.mainLayout)

	def onSearchButtonClicked(self):
		"""called when the search button is clicked"""
		print("search button clicked")
		self.searchResultsList.clear()
		self.searchResultsList.addItem("test1")
		self.searchResultsList.addItem("test2")

	def onSearchResultsSelectionChanged(self):
		"""called when the search results selection changes"""
		print("search results selection changed")
		print(self.searchResultsList.selectedItems())

