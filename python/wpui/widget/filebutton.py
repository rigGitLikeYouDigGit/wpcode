
"""buttons that often come up"""
import inspect, pprint
import typing as T
from pathlib import Path, PurePath
from PySide2 import QtCore, QtWidgets, QtGui

from wplib import TypeNamespace
from wpexp import EVAL
class FileSelectMode(TypeNamespace):
	"""should options look for a file or for a directory"""
	class _Base(TypeNamespace.base()):
		pass
	class Directory(_Base):pass
	class File(_Base):pass

class FileBrowserButton(QtWidgets.QPushButton):
	"""button to bring up file browser
	Ɏ 🂶 🂾 ㆔ ㅖ 〓 【 】 ☰ ☲ ☱
	"""

	pathSelected = QtCore.Signal(PurePath)

	Mode = FileSelectMode

	def __init__(self, name="☰", parent=None, defaultBrowserPath:T.Union[str, PurePath, T.Callable]="",
	             mode:Mode=Mode.File,
	             fileMask="",
	             browserDialogCls=QtWidgets.QFileDialog,
	             browserDialogCaption=""):
		super(FileBrowserButton, self).__init__(name, parent)
		self.defaultBrowserPath = defaultBrowserPath
		self.mode = mode
		self.fileMask = fileMask
		self.browserDialogCls = browserDialogCls
		self.browserDialogCaption = browserDialogCaption or "Select " + str(mode)

		self.pressed.connect(self.onPressed)
		self.setContentsMargins(0, 0, 0, 0)

		# do not allow horizontal stretching
		self.setSizePolicy(
			QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)

		# simple fn call
		self.setText(name)
		# get endpoints in by driving them
		rx(name).drive(self.setText)


	def sizeHint(self) -> QtCore.QSize:
		fontMetrics = QtGui.QFontMetrics(self.font())
		box = fontMetrics.size(
			QtCore.Qt.TextSingleLine, self.text())
		return box * 1.5


	def evalDefaultPath(self):
		return self.defaultBrowserPath if not \
			callable(self.defaultBrowserPath) else self.defaultBrowserPath()

	def onPressed(self):
		"""show file browser widget"""

		if self.mode == self.Mode.File:
			result = self.browserDialogCls.getOpenFileName(
				parent=self, caption=self.browserDialogCaption,
				dir=self.evalDefaultPath(),
				filter=self.fileMask
			)[0] or ""
		# elif self.mode == self.Mode.Directory:
		else:
			result = self.browserDialogCls.getExistingDirectory(
				parent=self, caption=self.browserDialogCaption,
				dir=self.evalDefaultPath(),
			) or ""
		if not result:
			return
		self.pathSelected.emit(PurePath(result))
		#return result
