from __future__ import annotations
import typing as T


from maya.api import OpenMaya as om, OpenMayaRender as omr, OpenMayaUI as omui, OpenMayaAnim as oma

"""lib for common attribute functions - bind, live mode, reset, etc"""

def makeBindAttr(name="bind")->(om.MObject, om.MFnEnumAttribute):
	eFn = om.MFnEnumAttribute()
	aBind = eFn.create(name, name, 0)
	eFn.addField("off", 0)
	eFn.addField("bind", 1)
	eFn.addField("bound", 2)
	eFn.addField("live", 3)
	return aBind, eFn


def makeBalanceWheelAttr(name="balanceWheel",
                         readable=True,
                         writable=True)->(om.MObject, om.MFnNumericAttribute):
	"""bool attribute to flag that a node has been eval'd
	like a message attribute that can be marked dirty
	"""
	nFn = om.MFnNumericAttribute()
	aBalanceWheel = nFn.create(name, name, om.MFnNumericData.kBoolean, 0)
	nFn.readable = readable
	nFn.writable = writable
	return aBalanceWheel

def jumpToElement(mArr:om.MArrayDataHandle, index, elementIsArray=False):
	"""jump to element in array plug"""
	if len(mArr) <= index:
		builder : om.MArrayDataBuilder = mArr.builder()
		if elementIsArray:
			builder.addElement(index)
		else:
			builder.addElementArray(index)

		mArr.set(builder)
	mArr.jumpToPhysicalElement(index)

def initArray(mArr:om.MArrayDataHandle, useInputValue=False):
	jumpToElement(mArr, 0)
	if useInputValue:
		dh : om.MDataHandle = mArr.inputValue()
	else:
		dh : om.MDataHandle = mArr.outputValue()
	obj : om.MObject = dh.data()
	if obj.isNull():
		print("init empty array")
		dh.setMObject(om.MObject())

def readDHGeneral(dhSrc:om.MDataHandle)->tuple[(float, om.MObject), int]:
	"""extract data from data handle - either an mobject,
	or a numeric float"""
	try:
		obj : om.MObject = dhSrc.data()
	except RuntimeError:
		return dhSrc.asDouble(), dhSrc.numericType()

	if obj.isNull():
		return dhSrc.asDouble(), dhSrc.numericType()

	return om.MObject(obj), dhSrc.type()

def writeDHGeneral(dhDst:om.MDataHandle, data:tuple[(float, om.MObject), int]):
	"""write data to data handle - either an mobject,
	or a numeric float"""
	print("writeDH", dhDst, data, type(data[0]))
	if isinstance(data[0], float):
		dhDst.setDouble(data[0])
	else:
		dhDst.setMObject(data[0])



