
#pragma once
#include <vector>
//#include <array>
#include <maya/MMatrix.h>
#include <maya/MDrawRegistry.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MPointArray.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MDGModifier.h>
#include <maya/MString.h>

#include <maya/MFnDependencyNode.h>
#include <maya/MFnTransform.h>
#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnGenericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>
#include <maya/MFnNurbsCurve.h>


#include "../macro.h"
#include "../api.h"
#include "stratasurfacenode.h"
#include "../lib.h"


using namespace ed;


MTypeId StrataSurface::kNODE_ID(0x00122C2A);
MString StrataSurface::kNODE_NAME("StrataSurface");

MString StrataSurface::drawDbClassification("drawdb/geometry/StrataSurface");
MString StrataSurface::drawRegistrantId("StrataSurfacePlugin");


MObject StrataSurface::aBalanceWheel;
MObject StrataSurface::aStOutMesh;

//MObject StrataSurface::aStEditMode;
MObject StrataSurface::aStUiData;

void StrataSurface::postConstructor() {

}

MStatus StrataSurface::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    //inputs
   

    // outputs
    aBalanceWheel = nFn.create("balanceWheel", "balanceWheel",
        MFnNumericData::kBoolean, 0);
    nFn.setWritable(false);
    nFn.setCached(false);

    aStOutMesh = tFn.create("stOutMesh", "stOutMesh",
        MFnData::kMesh);
    tFn.setWritable(false);
    tFn.setCached(true);

    //aStEditMode = nFn.create("stEditMode", "stEditMode", MFnNumericData::kBoolean, 0);
    nFn.setChannelBox(1);
    aStUiData = tFn.create("stUiData", "stUiData", MFnData::kString);


    std::vector<MObject> driverObjs = {
        //aStEditMode, 
        aStUiData,
 
    };

    std::vector<MObject>drivenObjs = {
        aBalanceWheel,
        aStOutMesh,
    };

    //addAttributes<StrataSurface>(driverObjs);
    //addAttributes<StrataSurface>(drivenObjs);

    setAttributesAffect<StrataSurface>(driverObjs, drivenObjs);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataSurface::compute(const MPlug& plug, MDataBlock& data) {
    MStatus s = MS::kSuccess;

    if (data.isClean(plug)) {
        return s;
    }
    DEBUGS("surface compute");

    // flip balancewheel
    data.outputValue(aBalanceWheel).setBool(!data.outputValue(aBalanceWheel).asBool());
    data.setClean(aBalanceWheel);

    data.setClean(plug);
    return s;
}



