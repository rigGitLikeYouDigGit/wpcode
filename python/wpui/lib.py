
from __future__ import annotations
import typing as T

import numpy as np
import pathlib

from PySide2 import QtWidgets, QtCore, QtGui

from wplib.sequence import toSeq

import subprocess

def openExplorerOnPath(path:(pathlib.Path, str),
                       lowestExisting=True):
	"""opens a new process of windows explorer,
	focused on given path
	"""
	p = pathlib.Path(path)
	if not p.exists():
		if lowestExisting:
			while not p.exists():
				try: p = p.parents[0]
				except IndexError:
					raise FileNotFoundError(f"No existing path to open explorer for any part of path {path}, {type(path)} ")
		else:
			raise FileNotFoundError(f"No existing path to open explorer for path {path}, {type(path)} ")
	if p.is_file():
		return subprocess.Popen(f'explorer /select,"{p}"')
	elif p.is_dir():
		return subprocess.Popen(f'explorer "{p}"')

def widgetParents(w:QtWidgets.QWidget):
	result = []
	while w.parentWidget():
		result.append(w.parentWidget())
		w = w.parentWidget()
	return result

def rootWidget(w:QtWidgets.QWidget):
	while w.parentWidget():
		w = w.parentWidget()
	return w

def widgetChildMap(w:QtWidgets.QWidget, includeObjects=True, onlyNamed=True)->dict[str, (QtWidgets.QWidget, QtCore.QObject)]:
	result = {}
	for i in w.children():
		if not isinstance(i, QtWidgets.QWidget):
			if not includeObjects:
				continue
		if not str(i.objectName()):
			if not onlyNamed:
				continue
		result[i.objectName()] = i
	return result

class muteQtSignals:
	"""small context for muting qt signals around a block"""
	def __init__(self, obj:QtCore.QObject):
		self.objs = toSeq(obj)
	def __enter__(self):
		for i in self.objs:
			i.blockSignals(True)
	def __exit__(self, exc_type, exc_val, exc_tb):
		for i in self.objs:
			i.blockSignals(False)

def arrToQMatrix(arr:np.ndarray)->QtGui.QMatrix:
	"""convert numpy array to QMatrix"""
	return QtGui.QMatrix(arr[0, 0], arr[0, 1], arr[1, 0], arr[1, 1], arr[0, 2], arr[1, 2])

def qmatrixToArr(mat:QtGui.QMatrix)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])

def qTransformToArr(mat:QtGui.QTransform)->np.ndarray:
	"""convert QMatrix to numpy array"""
	return np.array([[mat.m11(), mat.m12(), mat.dx()],
	                 [mat.m21(), mat.m22(), mat.dy()],
	                 [0, 0, 1]])

def qRectToArr(rect:(QtCore.QRect, QtCore.QRectF),
               originSize=True)->np.ndarray:
	dtype = float
	if isinstance(rect, QtCore.QRect):
		dtype = int
	if originSize:
		arr = np.array([rect.topLeft().toTuple(), rect.size().toTuple()], dtype=dtype)
	else:
		arr = np.array([rect.topLeft().toTuple(), rect.bottomRight().toTuple()], dtype=dtype)
	return arr

