
from __future__ import annotations
import typing as T

import numpy as np
from scipy.spatial import transform
from scipy.spatial.transform import Rotation, Slerp, RotationSpline
from scipy.interpolate import make_interp_spline, interp1d


np.set_printoptions(precision=3, suppress=True)
# from maya import cmds
# import maya.api.OpenMaya as om
#
# from wpm.core import getMObject#, getMFn



def setWorldMatrix():
	pass

def rotMatrix(mat:np.ndarray):
	return mat[:3, :3]

def normalized(a, axis=-1, order=2):
	"""courtesy of messr
	Eelco Hoogendoorn
	on SO
	"""
	a = np.array(a)
	l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
	l2[l2==0] = 1
	if len(a.shape) == 1:
		return (a / np.expand_dims(l2, axis))[0]
	return a / np.expand_dims(l2, axis)

def makeMat4(x=(1, 0, 0),
            y=(0, 1, 0),
            z=(0, 0, 1),
            w=(0, 0, 0),
            )->np.ndarray:
	arr = np.zeros((4, 4))
	arr[0, :3] = normalized(x[:3])
	arr[1, :3] = normalized(y[:3])
	arr[2, :3] = normalized(z[:3])
	arr[3, :len(w)] = w
	return arr

def matricesBetween(startMat, endMat,
        paramRange:(int, T.Iterable[int])=np.linspace(0.001, 0.999, 5)
                    )->np.ndarray:
	"""assume posX axis is tangent, Z up -
	X along axis, slerp Z from startMat to endMat.
	"""
	if isinstance(paramRange, int):
		paramRange = np.linspace(0.001, 0.999, paramRange)

	startPos = startMat[3, :3]
	endPos = endMat[3, :3]

	tangent = normalized(endPos - startPos)
	# endY = np.cross(tangent, endMat[2, :3])
	# endZ = np.cross(tangent, endY)
	# endRotMat = np.array([tangent, endY, endZ])
	# print("end rot mat", endRotMat)

	posInterpolator = interp1d(
		[0.0, 1.0],
		[startPos, endPos],
		axis=0
	)
	positions = posInterpolator(paramRange)
	upInterpolator = interp1d(
		[0.0, 1.0],
		[startMat[2, :3], endMat[2, :3]],
		axis=0
	)
	seedZVectors = normalized(upInterpolator(paramRange))
	yVectors = np.cross([tangent], seedZVectors)
	zVectors = np.cross([tangent], yVectors)

	mats = np.array([np.identity(4) for i in paramRange])

	mats[:, 0, :3] = [tangent]
	mats[:, 1, :3] = yVectors
	mats[:, 2, :3] = zVectors
	mats[:, 3, :3] = positions
	return mats





if __name__ == '__main__':

	r = np.linspace([0.0, 2.0, 0.0],
	                [4.0, 3.0, 10.0],
	                11
	                )
	#print(r)

	startMat = makeMat4()

	endMat = makeMat4(x=(0, 1, 0),
	                  y=(0, 0, 1),
	                  z=(1, 0, 0),
	                  w=(4, 0, 4))

	mats = matricesBetween(startMat, endMat, 4)
	print(mats)




