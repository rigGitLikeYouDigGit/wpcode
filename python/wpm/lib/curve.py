from __future__ import annotations
import types, typing as T
import pprint

from wplib import log
from wplib.sequence import flatten
from wpm import om, cmds, getMObject
import numpy as np
import scipy.interpolate as scinterp

"""
TODO:
move some of the pure-maths nurbs to a shared library above dcc level?
"""

unsignedIntervalArray = lambda n : np.linspace(0.0, 1.0, n)

def normaliseVector(v):
	return v / np.sqrt(np.sum(v ** 2))
def normaliseVectorArray(varr):
	return np.array([normaliseVector(i) for i in varr])

def uniformCurveParamInterpolator(mfn:om.MFnCurve, resolution=30)->scinterp.interp1d:
	"""maya api methods rarely take in 0-1 curve param -
	returns an interpolator for range
	[0, 1]  ->  [0, nSpans]
	"""
	uniformArray = np.linspace(0.0, 1.0, resolution)
	# spans = mfn.numSpans
	# degree = mfn.degree
	minParam, maxParam = mfn.knotDomain
	paramArray = np.linspace(minParam, maxParam, resolution)
	return scinterp.interp1d(uniformArray, paramArray)


def uniformToArcLengthInterpolator(mfn, resolution=30)->scinterp.interp1d:
	"""return interpolator mapping a uniform parametre to
	corresponding arclength parametre

	for each param, keep sum of all linear distances,
	as arclength param
	"""
	paramInterpolator = uniformCurveParamInterpolator(mfn, resolution)
	coordArray = np.linspace(start=0.0, stop=1.0, num=resolution)
	lengthArray = np.zeros((resolution))

	prevPoint = mfn.getPointAtParam(0.0)

	for i, param in enumerate(coordArray):
		point = mfn.getPointAtParam(
			# expand from 0-1 to 0-nSpans
			paramInterpolator(param)
		)
		length = (point - prevPoint).length()
		lengthArray[i] = length

	return scinterp.interp1d(coordArray, lengthArray, kind="quadratic")

def knotArrayForCurve(nCvs:int, degree=2):
	"""number of knots is (degree + N - 1)

	multiplicity of a knot = how many times a knot is duplicate. 3 values in a row is a multiplicity of 3 - multiplicity must be at max the degree of the curve, and such
	causes a sharp point
	"""
	if degree == 1 :
		return np.array([i for i in range(nCvs)])
	arr = np.zeros(degree + nCvs - 1)
	i = 0
	v = 0
	for n in range(degree):
		arr[i] = v
		i += 1
	for n in range(nCvs - degree):
		v += 1
		arr[i] = v
		i += 1
	for n in range(degree-1):
		arr[i] = v
		i += 1
	return arr


def curvePositionArray(
		mfn, coordArray:np.array=None, steps=None)->np.array:
	"""2d position array along curve -
	return [n, 3] array, converting points to vectors"""
	if steps:
		coordArray = np.linspace(0.0, 1.0, steps)
	positionArr = np.zeros((len(coordArray), 3))
	for i, coord in enumerate(coordArray):
		positionArr[i] = tuple(mfn.getPointAtParam(coord))[:-1]
	return positionArr

def curveTangentArray(mfn, coordArray:np.array=None, steps=None)->np.array:
	"""would be faster and more elegant to do actual
	maths, but would take way longer to account for cases"""
	#return np.gradient(positionArray)
	if steps:
		print("steps")
		coordArray = unsignedIntervalArray(steps)
	tanArr = np.zeros((len(coordArray), 3))
	tanArr[0] =  mfn.getDerivativesAtParam(0.0001)[1].normalize()
	#tanArr[0] = tuple(mfn.getPointAtParam(0.0))[:-1]
	for i, coord in enumerate(coordArray):
		if not i:
			continue
		tanArr[i] = (mfn.getPointAtParam(coord) - mfn.getPointAtParam(coord - 0.0001))
		# tanArr[i] = mfn.getDerivativesAtParam(
		#     coord, space=om.MSpace.kWorld)[1]
	#tanArr = normaliseVectorArray(tanArr)
	return tanArr


def makeRMF(posArray, tanArray, initVector=(1.0, 0, 0))->np.array:
	"""arrays should match desired coord points
	creates array of normals forming rotation-minimising frame
	along curve

	directly copied from
	Computation of Rotation Minimizing Frames -
	Wang, Juttler, Zheng, Liu
	DOI = 10.1145/1330511.1330513
	"""

	mv = om.MVector
	normalArray = np.zeros((len(posArray), 3))
	tanArray = normaliseVectorArray(tanArray)
	prevPos = mv(posArray[0])
	prevTan = mv(tanArray[0])
	prevNormal = normalArray[0] = mv(np.cross(prevTan, initVector)).normalize()

	for i, pos in enumerate(posArray[:-1]):
		"""use double-reflection method for rmf"""
		v1 = posArray[i+1] - posArray[i]
		c1 = np.dot(v1, v1)
		prevNormal = normalArray[i]
		rLi = prevNormal - (2.0 / c1) * (np.dot(v1, prevNormal)) * v1
		prevTan = tanArray[i]
		tLi = prevTan - (2.0 / c1) * (np.dot(v1, prevTan)) * v1

		v2 = tanArray[i + 1] - tLi
		c2 = np.dot(v2, v2)
		nextNormal = rLi -(2.0 / c2) * (np.dot(v2, rLi)) * v2

		normalArray[i + 1] = nextNormal

	# debug cubes
	debug = False # works fine
	if debug:
		for pos, tan, normal in zip(posArray, tanArray, normalArray):
			binormal = np.cross(tan, normal)
			matArgs = flatten([
				*tan, 0.0,
				*normal, 0.0,
				*binormal, 0.0,
				*pos, 1.0])
			mat = om.MMatrix(matArgs)
			testCube = cmds.polyCube()[0]
			mfn = om.MFnTransform(getMObject(testCube))
			mfn.setTransformation(
				om.MTransformationMatrix(mat)
			)


	return normalArray








