
from __future__ import annotations
import typing as T
from dataclasses import dataclass

from maya.api import OpenMaya as om

from wpm.lib.plugin import attr

@dataclass
class SolverFrameData:
	"""data for a single frame of the solver"""
	atFrame : int # frame number for this data
	float : tuple[float] = ()

def getSolverFrameData(
		parentDH:om.MDataHandle,
		atFrame:int,
		floatMObject:om.MObject,

		getOutputValues=True,
		)->SolverFrameData:
	"""extract the solver frame data from the given data handle"""
	def _getLeafDH(dh)->om.MDataHandle:
		if getOutputValues:
			return dh.outputValue()
		else:
			return dh.inputValue()
	floatArr = om.MArrayDataHandle(parentDH.child(floatMObject))
	floats = []
	for i in range(len(floatArr)):
		attr.jumpToElement(floatArr, i)
		floats.append(float(_getLeafDH(floatArr).asDouble()))

	return SolverFrameData(
		atFrame=atFrame,
		float=tuple(floats))


def setSolverFrameData(
		parentDH:om.MDataHandle,
		solverFrameData:SolverFrameData,
		floatMObject:om.MObject,
):
	# floats
	floatArr = om.MArrayDataHandle(parentDH.child(floatMObject))
	for i, f in enumerate(solverFrameData.float):
		attr.jumpToElement(floatArr, i)
		floatArr.outputValue().setDouble(f)



def makeFrameCompound(name:str,

					floatArrName="float",
                      readable:bool=True, writable:bool=True,
                      array:bool=False,
                      )->dict[str, om.MObject]:
	"""make a compound attribute for a single frame of data"""
	cFn = om.MFnCompoundAttribute()
	compoundObj = cFn.create(name, name)
	cFn.readable = readable
	cFn.writable = writable
	if array:
		cFn.array = True
		cFn.usesArrayDataBuilder = True

	# make float attribute
	nFn = om.MFnNumericAttribute()
	floatObj = nFn.create(floatArrName, floatArrName, om.MFnNumericData.kDouble, 0.0)
	nFn.readable = readable
	nFn.writable = writable
	nFn.array = True
	nFn.usesArrayDataBuilder = True

	cFn.addChild(floatObj)

	return {
		"compound" : compoundObj,
		"float" : floatObj,
	}

