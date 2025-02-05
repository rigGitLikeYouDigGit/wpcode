
#pragma once
#include "api.h"

#include "stratapoint.h"

using namespace ed;

MTypeId StrataPoint::kNODE_ID(0x00122C1C);
MString StrataPoint::kNODE_NAME("strataPoint");

MObject StrataPoint::aEditMode;

MStatus StrataPoint::initialize() {
	MStatus s = MS::kSuccess;
	MFnNumericAttribute nFn;
	aEditMode = nFn.create("editMode", "editMode", MFnNumericData::kBoolean, 0);

	addAttribute(aEditMode);
	

	CHECK_MSTATUS_AND_RETURN_IT(s);
	return s;
}