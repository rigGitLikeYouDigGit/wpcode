
from __future__ import annotations
import typing as T

from PySide2 import QtWidgets, QtCore, QtGui

from maya.app.general.mayaMixin import MayaQWidgetDockableMixin as MQ
from shiboken2 import wrapInstance


from wp.ui.widget import WpWidgetBase
from wpm import cmds, om, oma, omui, WN, createWN

"""functions for interfacing with maya's ui"""

def getMayaWindow()->QtWidgets.QMainWindow:
	"""return maya's main window"""
	from maya import OpenMayaUI as omui
	mayaMainWindowPtr = omui.MQtUtil.mainWindow()
	mayaMainWindow = wrapInstance(int(mayaMainWindowPtr), QtWidgets.QWidget)
	return mayaMainWindow

def iterAllQObjects(parent:QtWidgets.QWidget)->T.Iterator[QtWidgets.QWidget]:
	"""iterate over all qobjects in widget tree"""
	for child in parent.children():
		yield child
		yield from iterAllQObjects(child)

def findGlobalQObjectByName(name:str, parent=None)->T.Optional[QtWidgets.QWidget]:
	"""find global object by name"""
	parent = parent or getMayaWindow()
	for i in iterAllQObjects(parent):
		if i.objectName() == name:
			return i
	return None

class AllEventFilter(QtCore.QObject):
	"""filter all events back to maya"""
	def eventFilter(self, obj, event):
		#print(obj, event)
		return False


class ToolDockWindow(
	MQ, QtWidgets.QWidget, WpWidgetBase
                     ):
	"""outer widget class letting other widgets dock to maya UI
	"""

	strPrefix = "toolWindow_"

	def __init__(self,
	             innerWidget:QtWidgets.QWidget,
	             parent=None):
		"""create dockable window with inner widget
		:param innerWidget: widget to dock
		:param parent: parent widget

		for now we only allow one instance, if you want named
		tool windows or something, you'll have to do it yourself
		"""
		parent = parent or getMayaWindow()
		super(ToolDockWindow, self).__init__(parent=parent)
		WpWidgetBase.__init__(self)
		innerWidget.setParent(self)
		self._innerWidget = innerWidget
		self.setWindowTitle(innerWidget.objectName())
		self.setObjectName(
			self.toolWindowObjectNameForWidget(innerWidget)
		)
		vl = QtWidgets.QVBoxLayout()
		vl.addWidget(innerWidget)
		self.setLayout(vl)

		# filter all events back to maya
		self.installEventFilter(AllEventFilter())

		# dock windows should be ui roots
		self.isWidgetRoot = True


	def innerWidget(self):
		return self.layout().itemAt(0).widget()

	def deleteExisting(self):
		"""delete existing window with same name.
		Happily we can just use cmds for this
		"""
		if cmds.window(self.windowTitle(), exists=True):
			cmds.deleteUI(self.windowTitle())

	@classmethod
	def toolWindowObjectNameForWidget(cls, widget:QtWidgets.QWidget):
		"""return object name for tool window for widget class
		"""
		return cls.strPrefix + widget.objectName()

	@classmethod
	def findExistingWindowForWidget(cls
	                                , widget:QtWidgets.QWidget):
		"""find existing window for widget class
		"""
		mainWin = getMayaWindow()
		return findGlobalQObjectByName(
			cls.toolWindowObjectNameForWidget(widget)
		)


	@classmethod
	def getToolWindowForWidget(cls, widget:QtWidgets.QWidget,
	                  deleteExisting=True):
		"""get new or existing tool window for widget class
		call this"""
		winFound = cls.findExistingWindowForWidget(widget)

		if not winFound:
			return cls(widget)
		if not deleteExisting:
			return winFound

		# dock widgets always have a WorkspaceControl widget above
		ctl = findGlobalQObjectByName(
			cls.toolWindowObjectNameForWidget(widget) + "WorkspaceControl"
		)

		cmds.deleteUI(ctl.objectName()) # deleteUI seems to take care of everything

		return cls(widget)

	def show(self, dockable=True, floating=True, area='left'):
		"""show dockable window"""
		return MQ.show(
			self, dockable=dockable, floating=floating, area=area,
			retain=False
		)

	def close(self):
		"""close window - fully delete it for now"""
		print("tool window close")
		workspace = self.parent()
		super(ToolDockWindow, self).close()
		cmds.deleteUI(workspace.objectName())
		workspace.setParent(None)
		workspace.deleteLater()

	def closeEvent(self, event:PySide2.QtGui.QCloseEvent) -> None:
		""""""
		print("tool window close event")
		super(ToolDockWindow, self).closeEvent(event)

	def hide(self) -> None:
		"""there's some really weird stuff about hiding vs actually closing -
		maya natively only ever hides docked windows, which is
		often a pain.

		Here we send a close signal to the innerwidget and unparent it -
		restarting it should attach a NEW inner widget to the PERSISTENT
		tool window, preserving its spot in the maya ui, etc
		"""
		print("tool window hide")
		super(ToolDockWindow, self).hide()
		self.innerWidget().setParent(None)
		self.innerWidget().close()
		self.innerWidget().deleteLater()

	def hideEvent(self, event:PySide2.QtGui.QHideEvent) -> None:
		""""""
		print("tool window hide event")
		super(ToolDockWindow, self).hideEvent(event)
		self._innerWidget.setParent(None)
		self._innerWidget.close()
		self._innerWidget.deleteLater()
		self._innerWidget = None


def showToolWindow(widget:QtWidgets.QWidget,
                   deleteExisting=True,
                   dockable=True,
                   floating=True,
                   area='left'
)->ToolDockWindow:
	"""show tool window for widget class"""
	toolWin = ToolDockWindow.getToolWindowForWidget(
		widget, deleteExisting=deleteExisting)
	toolWin.show(dockable=dockable, floating=floating, area=area)
	return toolWin
