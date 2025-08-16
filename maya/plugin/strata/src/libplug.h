#pragma once

#include <maya/MObject.h>
#include <maya/MPlug.h>
#include <maya/MDataHandle.h>
#include <maya/MMatrix.h>
#include <maya/MMatrixArray.h>
#include <maya/MFnMatrixData.h>
#include <maya/MDoubleArray.h>
#include <maya/MFnTypedAttribute.h>

#include "macro.h"


namespace strata {

	inline MMatrix accessMMatrixDH(MDataHandle& dh) {
		// get MMatrix out of a plug, without running the risk of crashing
		return MFnMatrixData(dh.data()).matrix();
	}
}

