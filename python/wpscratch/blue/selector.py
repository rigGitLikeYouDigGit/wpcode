from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from __future__ import annotations
import typing as T, types
import importlib, traceback
from pathlib import Path

from Qt import QtCore, QtWidgets, QtGui

from ..lib import linux
from u_rig.tools.rig_module.rig_module import RigModule

from ..asset import Asset

if T.TYPE_CHECKING:
	from ..plan import Plan


class AssetSelector(QtWidgets.QWidget):
	assetChanged = QtCore.Signal(list)

	def __init__(self, parent=None):
		super().__init__(parent)

		self.show_box = QtWidgets.QComboBox(self)
		self.type_box = QtWidgets.QComboBox(self)
		self.asset_box = QtWidgets.QComboBox(self)
		self.rig_box = QtWidgets.QComboBox(self)
		self.rig_box.setEditable(True)

		self.menus = [self.show_box, self.type_box, self.asset_box, self.rig_box]

		self.scriptPathLine = QtWidgets.QLineEdit(self)
		self.scriptPathLine.setReadOnly(True)  # don't allow setting from path for now
		self.scriptPathLine.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)

		# self.refresh_btn = QtWidgets.QPushButton("refresh", self)

		self.make_connections()
		self.make_layout()

		self.populate_menus()

	def path_line_ctx_requested(self, pos):
		menu = QtWidgets.QMenu()
		explorerAction = menu.addAction("open dir")
		explorerAction.triggered.connect(lambda *args, **kwargs:
		                                 linux.open_caja_at_dir(
			                                 Path(self.scriptPathLine.text()).parent))
		menu.exec_(self.scriptPathLine.mapToGlobal(pos))

	def make_layout(self):
		vl = QtWidgets.QVBoxLayout(self)
		hl = QtWidgets.QHBoxLayout(self)
		hl.setContentsMargins(1, 1, 1, 1)
		# hl.addWidget(self.refresh_btn)
		for i in self.menus:
			hl.addWidget(i)
		vl.addLayout(hl)
		vl.addWidget(self.scriptPathLine)
		vl.setContentsMargins(0, 0, 0, 0)
		self.setLayout(vl)

	def get_tokens(self):
		return [self.show_box.currentText(), self.type_box.currentText(), self.asset_box.currentText()]

	def make_connections(self):
		for i in self.menus[:3]:
			i.currentTextChanged.connect(self._on_box_changed)
		self.scriptPathLine.customContextMenuRequested.connect(self.path_line_ctx_requested)

	def setTokens(self, tokens: list[str]):
		"""set as many as given, auto manage the rest"""
		for i, v in enumerate(tokens):
			self.menus[i].setCurrentText(v)
		self.populate_menus(len(tokens))

	def get_asset(self) -> Asset:
		return Asset(self.show_box.currentText(), self.type_box.currentText(), self.asset_box.currentText())

	def get_rig_name(self):
		return self.rig_box.currentText()

	def get_rig_path(self) -> Path:
		print("")
		if not self.get_asset().exists():
			return None
		return self.get_asset().scriptDir() / (self.get_rig_name() + ".plan.json")  # I think this is easiest

	def get_prev_asset_tokens(self):
		"""TODO: save prev asset in pref folder or something"""
		return ["", "", ""]

	build_asset_types = ["character", "vehicle", "prop"]

	def populate_menus(self, fromIndex=0):
		"""could refactor this by doing full asset paths later"""
		self.blockSignals(True)
		for i in self.menus:
			i.blockSignals(True)
			i.setEnabled(True)
		prev_asset = self.get_prev_asset_tokens()
		if (fromIndex < 1):
			self.show_box.clear()
			show_map = Asset.showDirMap()
			for k, v in sorted(show_map.items()):
				self.show_box.addItem(k)
			if (prev_asset[0] in show_map):
				self.show_box.setCurrentText(prev_asset[0])
		current_show = self.show_box.currentText()
		current_show_dir = Asset.showDirMap()[current_show]

		# if build dir doesn't exist, halt
		if not (current_show_dir / "build").exists():
			for m in self.menus[1:]:
				m.clear()
				m.addItem("no build dir")
				m.setCurrentIndex(0)
				m.setEnabled(False)
			self.blockSignals(False)
			for i in self.menus:
				i.blockSignals(False)
			return

		if (fromIndex < 2):
			self.type_box.clear()
			type_map = Asset.assetTypeDirMap(current_show_dir)
			for i in self.build_asset_types:
				if i in type_map:
					self.type_box.addItem(i)
			# for k, v in sorted(type_map.items()):
			#     self.type_box.addItem(k)
			if (prev_asset[1] in type_map):
				self.type_box.setCurrentText(prev_asset[1])
		current_type = self.type_box.currentText()
		current_type_dir = Asset.assetTypeDirMap(current_show_dir)[current_type]
		if (fromIndex < 3):
			self.asset_box.clear()
			asset_map = Asset.assetDirMap(current_type_dir)
			found_asset = False
			for k, v in sorted(asset_map.items()):
				if (k.startswith(".")):
					continue
				self.asset_box.addItem(k)
				found_asset = True
			if not found_asset:
				for i in self.menus[2:]:
					i.clear()
					i.addItem("no assets")
					i.setEnabled(False)
				self.blockSignals(False)
				for i in self.menus:
					i.blockSignals(False)
				return

			if (prev_asset[2] in asset_map):
				self.asset_box.setCurrentText(prev_asset[2])
		current_asset = self.asset_box.currentText()
		current_asset_dir = Asset.assetDirMap(current_type_dir)[current_asset]

		asset = self.get_asset()
		self.rig_box.clear()
		self.rig_box.addItem("rig")

		this_index = self.asset_box.currentIndex()
		self.asset_box.setCurrentIndex(0)
		self.blockSignals(False)
		for i in self.menus:
			i.blockSignals(False)
		self.asset_box.setCurrentIndex(this_index)

	def _on_box_changed(self, *args, **kwargs):
		i = self.menus.index(self.sender())
		try:
			self.populate_menus(i + 1)
		except Exception as e:
			self.blockSignals(False)
			for i in self.menus:
				i.blockSignals(False)
			traceback.print_exc()

		if self.get_asset().exists():
			self.scriptPathLine.setText(str(self.get_rig_path()))
		else:
			self.scriptPathLine.setText("<no asset dir found>")
		self.assetChanged.emit(self.get_tokens())