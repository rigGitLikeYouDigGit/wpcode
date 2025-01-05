from __future__ import annotations
import types, typing as T
import pprint
from wplib import log

from PySide2 import QtCore, QtGui, QtWidgets


def rootWindow():
	'''Returns the currently active QT main window
		Works for any Qt UI, like Maya, or now Max.
	Parameters

	thank you based tfox_td
	'''
	# for MFC apps there should be no root window
	window = None
	if QtWidgets.QApplication.instance():
		inst = QtWidgets.QApplication.instance()
		window = inst.activeWindow()
		# Ignore QSplashScreen's, they should never be considered the root window.
		if isinstance(window, QtWidgets.QSplashScreen):
			return None
		# If the application does not have focus try to find A top level widget
		# that doesn't have a parent and is a QMainWindow or QDialog
		if window == None:
			windows = []
			dialogs = []
			for w in QtWidgets.QApplication.instance().topLevelWidgets():
				if w.parent() == None:
					if isinstance(w, QtWidgets.QMainWindow):
						windows.append(w)
					elif isinstance(w, QtWidgets.QDialog):
						dialogs.append(w)
			if windows:
				window = windows[0]
			elif dialogs:
				window = dialogs[0]

		# grab the root window
		if window:
			while True:
				parent = window.parent()
				if not parent:
					break
				if isinstance(parent, QtWidgets.QSplashScreen):
					break
				window = parent

	return window