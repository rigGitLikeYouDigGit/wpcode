from __future__ import annotations
import types, typing as T
import pprint
#from wplib import log


from PySide6 import QtCore, QtWidgets, QtGui

import hou

""" this is surely a terrible idea, but it bothers me that we can't use
custom widgets in houdini node parametres


ok so after some investigation this might just be impossible, houdini UI 
panels don't map directly to Qt widgets, and drawing inside them is handled in GL.
Getting all children of the main window just gives this:
<PySide6.QtWidgets.QWidget(0x3a15ca40, name="RE_Window") at 0x0000000121451B80>
<PySide6.QtWidgets.QWidget(0x3a15ca40, name="RE_Window") at 0x0000000121451B80>
<hutil.qt.info.window.NodeInfoWindow(0x138cc8580) at 0x000000011AA55500>
<PySide6.QtWidgets.QWidget(0x11a565a20, name="RE_GLDrawableWrapper") at 0x000000017F08E740>
<PySide6.QtWidgets.QWidget(0x78fbdbc0, name="RE_Window") at 0x00000001198DB740>
<PySide6.QtWidgets.QLayout(0x138d9ab00, name = "_layout") at 0x000000011960EE40>
<PySide6.QtGui.QAction(0x1779036c0 text="Save Current Window Width as Default" toolTip="Save Current Window Width as Default" menuRole=TextHeuristicRole enabled=true visible=true) at 0x000000012126C6C0>
<PySide6.QtWidgets.QWidget(0x11a566860, name="RE_GLDrawable") at 0x000000017C60B7C0>
<PySide6.QtWidgets.QVBoxLayout(0xdf6c2220) at 0x000000011FAADC80>
<PySide6.QtWidgets.QVBoxLayout(0x790fcce0) at 0x0000000119651480>
<PySide6.QtWidgets.QWidget(0x1775a1000, name="RE_GLDrawableWrapper") at 0x0000000119653500>
<PySide6.QtWidgets.QWidget(0x1390aca40, name="RE_Window") at 0x0000000119651600>
<PySide6.QtWidgets.QWidget(0x1775a1090, name="RE_GLDrawable") at 0x0000000110E683C0>
<PySide6.QtWidgets.QVBoxLayout(0x17759b4c0) at 0x00000000D06C4180>
<PySide6.QtWidgets.QVBoxLayout(0x1390b7540) at 0x000000017F08E740>
<PySide6.QtWidgets.QWidget(0x13909f430, name="RE_GLDrawableWrapper") at 0x000000017C60B7C0>
<PySide6.QtWidgets.QWidget(0x13909f370, name="RE_GLDrawable") at 0x00000000D06C4180>
<PySide6.QtWidgets.QVBoxLayout(0x1390b7280) at 0x000000011FAADC80>
<PySide6.QtWidgets.QWidget(0x139092e80, name="RE_Window") at 0x000000017F08E740>
<PySide6.QtWidgets.QVBoxLayout(0x139035280) at 0x00000001198DB740>
<PySide6.QtWidgets.QWidget(0x13903a800, name="RE_GLDrawableWrapper") at 0x0000000110E683C0>
<PySide6.QtWidgets.QWidget(0x13903a740, name="RE_GLDrawable") at 0x000000017C60B7C0>
<PySide6.QtWidgets.QVBoxLayout(0x139034fc0) at 0x000000011FAADC80>
 <PySide6.QtWidgets.QVBoxLayout(0x139034fc0) at 0x000000011FAADC80>
 
 this may be one to raise with sidefx to see if there are any plans to redo the ui
"""


class CustomHoudiniParmWidget(QtWidgets.QWidget):

	def __init__(self, parent=None):
		super().__init__(parent)

		self.l = QtWidgets.QLabel("CUSTOM widget :D")
		self.setLayout(QtWidgets.QVBoxLayout())
		self.layout().addWidget(self.l)


def test():

	paramWs = []
	for i in hou.ui.paneTabs():
		if isinstance(i, hou.ParameterEditor):
			paramWs.append(i)
	print(paramWs)

	for i in paramWs:
		q = i.qtParentWindow()
		print(q)
		highlightWidget(q)
		#q.hide() # hides the entire houdini window

def findNodeParamWidget():
	rootWindow = lib.rootWindow()
	print(rootWindow)
	for w in QtWidgets.QApplication.instance().topLevelWidgets():
		print(w, w.objectName())

	t = rootWindow.dumpObjectTree()
	print(t)

	found = rootWindow.findChildren(
		QtWidgets.QWidget,
		QtCore.QRegularExpression("*")
	)

	print(len(found))
	for i in found:
		print(i, type(i))

	pass


