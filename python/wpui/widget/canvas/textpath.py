
from __future__ import annotations

"""drawing text with qt is the pathway to unimaginable agony
- ed, circa ~2020
he wasn't wrong
"""

import numpy as np

from PySide2 import QtCore, QtGui, QtWidgets


def transformsAlongPath(nTransforms:int, path:QtGui.QPainterPath,
                  startLimit=0.2,
                  endLimit=0.8
                  )->tuple[list[QtCore.QPoint], list[float]]:
	"""return list of position and angle along 2d path
	"""
	step = (endLimit - startLimit) / nTransforms
	positions = []
	angles = []
	for i in range(nTransforms):
		param = startLimit + step * i
		positions.append(path.pointAtPercent(param))
		angles.append(path.angleAtPercent(param))
	return positions, angles


def textAdvances(text:str, font:QtGui.QFont)->np.ndarray:
	"""return lengths of each char"""
	lengths = np.full(len(text), 0.0)
	metrics = QtGui.QFontMetrics(font)
	for i, s in enumerate(text):
		lengths[i] = metrics.horizontalAdvance(s)
	return lengths

def sampleTransformsAlongPath(sampleParams:np.ndarray, path:QtGui.QPainterPath,
                  )->tuple[list[QtCore.QPoint], list[float]]:
	"""return list of position and angle along 2d path
	"""
	positions = []
	angles = []
	for i in sampleParams:
		positions.append(path.pointAtPercent(i))
		angles.append(path.angleAtPercent(i))
	return positions, angles



def drawTextAlongPath(text:str, path:QtGui.QPainterPath,
                      painter:QtGui.QPainter,
                      startLimit=0.2,
                      endLimit=0.8,
                      angleAdd=0):
	"""enforced monospace for now,
	maybe allow passing in some kind of numpy array resampling idk
	this still doesn't look great but sick of working on it"""
	offsets = textAdvances(text, painter.font())
	offsets *= 1.2
	fm = QtGui.QFontMetrics(painter.font())
	textLen = fm.horizontalAdvance(text)
	normalisedOffsets = offsets / textLen
	#print(normalisedOffsets, sum(normalisedOffsets))
	startLimit = 0.2,
	endLimit = 0.8

	lengthFraction = textLen / (path.length())
	#print("lengthf", lengthFraction)
	pathOffsets = normalisedOffsets * lengthFraction
	startLimit = (1.0 - lengthFraction) / 2
	endLimit = (1.0 + lengthFraction) / 2
	# if lengthFraction > 1.0:
	# 	startLimit = 0.0
	# 	endLimit = 1.0
	# else:
	# 	startLimit = (1.0 - lengthFraction) / 2.0
	# 	endLimit = (1.0 - lengthFraction) / 2.0

	#totalParams =
	#sampleParams *= ((endLimit - startLimit) / sum(sampleParams))
	#offsets /= textLen
	pathOffsets = np.cumsum(pathOffsets)
	pathOffsets += startLimit
	#sampleParams += startLimit
	#print(pathOffsets)

	#positions, angles = sampleTransformsAlongPath(pathOffsets, path)
	positions, angles = transformsAlongPath(len(text), path)
	#print(positions, angles)
	painter.save()

	for i, (char, position, angle) in enumerate(zip(text, positions, angles)):
		#painter.restore()
		painter.save()
		#painter.resetMatrix()
		painter.translate(position)
		painter.rotate(-angle + angleAdd)
		#painter.drawText(QtCore.QPoint(-pathOffsets[i], -pathOffsets[i]), char)
		painter.drawText(fm.boundingRectChar(char).bottomRight(), char)
		painter.restore()

	painter.restore()





