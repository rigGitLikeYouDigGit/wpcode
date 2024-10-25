
"""buttons that often come up"""
import inspect, pprint
import typing as T
from pathlib import Path, PurePath
from PySide2 import QtCore, QtWidgets, QtGui

from wplib import TypeNamespace

from wpdex import *

#from wpexp import EVAL
class FileSelectMode(TypeNamespace):
	"""should options look for a file or for a directory"""
	class _Base(TypeNamespace.base()):
		pass
	class Dir(_Base):pass
	class File(_Base):pass

class FileBrowserButton(QtWidgets.QPushButton):
	"""atom button to bring up file browser
	Ɏ 🂶 🂾 ㆔ ㅖ 〓 【 】 ☰ ☲ ☱

	ignore file filters for now, I think they could be more intuitive -
	from the qt docs:
	If you want to use multiple filters, separate each one with two semicolons. For example:
	"Images (*.png *.xpm *.jpg);;Text files (*.txt);;XML files (*.xml)"
	can't make this up

	"""

	pathSelected = QtCore.Signal(list)

	Mode = FileSelectMode

	def __init__(self, name="☰", parent=None, defaultBrowserPath:T.Union[str, PurePath, T.Callable]="",
	             mode:Mode=Mode.File,
	             pathCls=Path,
	             fileMask="",
	             browserDialogCls=QtWidgets.QFileDialog,
	             browserDialogCaption="",
	             allowMultiple=False
	             ):
		super(FileBrowserButton, self).__init__(name, parent)
		self.defaultBrowserPath = defaultBrowserPath
		self.mode = mode
		self.pathCls = pathCls
		self.fileMask = fileMask
		self.browserDialogCls = browserDialogCls
		self.browserDialogCaption = browserDialogCaption or "Select " + str(mode)
		self.allowMultiple = allowMultiple

		self.pressed.connect(self.onPressed)
		self.setContentsMargins(0, 0, 0, 0)

		# do not allow horizontal stretching
		self.setSizePolicy(
			QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
		self.setText(name)

	def sizeHint(self) -> QtCore.QSize:
		fontMetrics = QtGui.QFontMetrics(self.font())
		box = fontMetrics.size(
			QtCore.Qt.TextSingleLine, self.text())
		return box * 1.5

	def onPressed(self):
		"""show file browser widget

		behold the kibble-like shrapnel of reactive connections
		maybe in the future we can have a subclass that
		automatically wraps functions like this in decorators?
		evaling all arguments, etc
		"""

		log("caption", EVAL1(self.browserDialogCaption))
		mode = EVAL1(self.mode)
		log("mode", mode, mode == "dir")
		if mode == self.Mode.File:
			if EVAL1(self.allowMultiple):
				result = self.browserDialogCls.getOpenFileNames(
					parent=self,
					caption=str(EVAL1(self.browserDialogCaption)),
					dir=str(EVAL1(self.defaultBrowserPath)),
					filter=EVAL1(self.fileMask)
				) or ""
				log("result files", result)
				# returns ([list of string paths], filter str)
				result = result[0]

			else:
				result = self.browserDialogCls.getOpenFileName(
					parent=self,
					caption=str(EVAL1(self.browserDialogCaption)),
					dir=str(EVAL1(self.defaultBrowserPath)),
					filter=EVAL1(self.fileMask)
				) or ""
				log("result file", result)
				# returns (str file, filter str)
				if not result:
					return
				result = [result[0]]
				# result = [Path(i) for i in result]
				# self.pathSelected.emit(result)

		else:
			#if EVAL1(self.allowMultiple):
			result = self.browserDialogCls.getExistingDirectory(
				parent=self,
				caption=EVAL1(self.browserDialogCaption),
				dir=str(EVAL1(self.defaultBrowserPath)),
			) or ""
			log("result dir", result)
			# returns raw dir string
			if not result:
				return
			result = [result]
			# else:
			# 	raise RuntimeError("No multiple dirs yet, no easy QT method for it")
			# 	result = self.browserDialogCls.getExistingDirectory(
			# 		parent=self,
			# 		caption=EVAL1(self.browserDialogCaption),
			# 		dir=str(EVAL1(self.defaultBrowserPath)),
			# 	) or ""
			# 	log("result dir", result)
			# 	# returns raw dir string
			# 	if not result:
			# 		return
				#result = [result]
		result = [Path(i) for i in result]
		log("emit result", result)
		self.pathSelected.emit(result)
		#return result
