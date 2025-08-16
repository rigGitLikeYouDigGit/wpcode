
#pragma once
#include <vector>
#include <algorithm>
//#include <array>


#include "../macro.h"
#include "../api.h"
#include "../libplug.h"
#include "stratacurvenode.h"
#include "../lib.h"
#include "../libnurbs.h"


using namespace strata;


MTypeId StrataCurve::kNODE_ID(0x00122C1F);
MString StrataCurve::kNODE_NAME("strataCurve");

MString StrataCurve::drawDbClassification("drawdb/geometry/StrataCurve");
MString StrataCurve::drawRegistrantId("StrataCurvePlugin");


MObject StrataCurve::aStStartMatrix;
MObject StrataCurve::aStStartTangent;
MObject StrataCurve::aStStartUp;
    MObject StrataCurve::aStStartUpX;
    MObject StrataCurve::aStStartUpY;
    MObject StrataCurve::aStStartUpZ;
MObject StrataCurve::aStStartTangentScale;
MObject StrataCurve::aStStartAutoTangentBlend;

MObject StrataCurve::aStEndMatrix;
MObject StrataCurve::aStEndTangent;
MObject StrataCurve::aStEndUp;
MObject StrataCurve::aStEndTangentScale;
MObject StrataCurve::aStEndAutoTangentBlend;

MObject StrataCurve::aStNSpans;
MObject StrataCurve::aStDegree;

MObject StrataCurve::aStOutUpCurve;
MObject StrataCurve::aStOutCurve;
MObject StrataCurve::aBalanceWheel;

MObject StrataCurve::aStEditMode;
MObject StrataCurve::aStUiData;




void StrataCurve::postConstructor() {
    // create a nurbsCurve shape node under this transform, 
    // connect this node's outCurve attribute to it.
    // also do the upcurve?
    // nah, naming would get inconsistent, and upcurve not always needed
    
    MStatus s(MS::kSuccess);
    MFnNurbsCurve crvFn;
    MFnDagNode thisFn(thisMObject());
    MDGModifier dgMod;
    MPointArray pointArr;
    //pointArr.append(MPoint());
    //pointArr.append(MPoint());
    //MObject crvObj = crvFn.createWithEditPoints(
    //    pointArr, 1, MFnNurbsCurve::kOpen, false, true, true,       
    //    thisMObject());
    //crvFn.setObject(crvObj);

    //thisFn.setName("strataCurve#");
    //crvFn.setName("strataCurveShape#");

    //MPlug crvOutPlug = thisFn.findPlug(MString("stOutCurve"), false, &s);
    //CHECK_MSTATUS(s);
    ////DEBUGS("found crvOut mplug", crvOutPlug.name());
    //MPlug crvCreatePlug = crvFn.findPlug("create", false, &s);
    //CHECK_MSTATUS(s);
    ////DEBUGS("found crvCreate mplug", crvCreatePlug.name());
    //
    //dgMod.connect(crvOutPlug, crvCreatePlug);
    //dgMod.doIt();
}

MStatus StrataCurve::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    //inputs
    //// start
    aStStartMatrix = mFn.create("stStartMatrix", "stStartMatrix", MFnMatrixAttribute::kDouble);
    mFn.setDefault(MMatrix::identity);

    //const double yArr[3] = { 0.0, 1.0, 0.0 };

    aStStartUp = makeVectorAttr("stStartUpVector", nFn//, yArr
    );
    nFn.setChannelBox(true);
    nFn.setDefault(0.0, 1.0, 0.0); // y up

    aStStartTangent = makeVectorAttr("stStartTangentVector", nFn);
    nFn.setChannelBox(true);
    nFn.setDefault(1.0, 0.0, 0.0);
    aStStartTangentScale = nFn.create("stStartTangentScale", "stStartTangentScale",
        MFnNumericData::kFloat, 1.0);
    nFn.setChannelBox(true);
    aStStartAutoTangentBlend = nFn.create("stStartAutoTangentBlend", "stStartAutoTangentBlend",
        MFnNumericData::k2Float, 1.0);
    nFn.setChannelBox(true);
    nFn.setMin(0);
    nFn.setMax(1);

    //// end
    aStEndMatrix = mFn.create("stEndMatrix", "stEndMatrix", MFnMatrixAttribute::kDouble);
    mFn.setDefault(MMatrix::identity);
    aStEndUp = makeVectorAttr("stEndUpVector", nFn);
    nFn.setChannelBox(true);
    nFn.setDefault(0.0, 1.0, 0.0); // y up

    aStEndTangent = makeVectorAttr("stEndTangentVector", nFn);
    nFn.setChannelBox(true);
    nFn.setDefault(-1.0, 0.0, 0.0);
    aStEndTangentScale = nFn.create("stEndTangentScale", "stEndTangentScale",
        MFnNumericData::kFloat, 1.0);
    nFn.setChannelBox(true);
    aStEndAutoTangentBlend = nFn.create("stEndAutoTangentBlend", "stEndAutoTangentBlend",
        MFnNumericData::k2Float, 1.0);
    nFn.setChannelBox(true);
    nFn.setMin(0);
    nFn.setMax(1);

    // curve settings
    aStNSpans = nFn.create("stNSpans", "stNSpans",
        MFnNumericData::kInt, 10);
    nFn.setMin(1);
    nFn.setChannelBox(true);
    aStDegree = nFn.create("stDegree", "stDegree",
        MFnNumericData::kInt, 3);
    nFn.setMin(1);
    nFn.setChannelBox(true);

    // outputs
    aBalanceWheel = nFn.create("balanceWheel", "balanceWheel",
        MFnNumericData::kBoolean, 0);
    nFn.setWritable(false);
    nFn.setCached(false);

    aStOutCurve = tFn.create("stOutCurve", "stOutCurve",
        MFnData::kNurbsCurve,
        MObject::kNullObj);
    tFn.setWritable(false);
    //tFn.setCached(true);
    
    aStOutUpCurve = tFn.create("stOutUpCurve", "stOutUpCurve",
        MFnData::kNurbsCurve,
        MObject::kNullObj);
    tFn.setWritable(false);
    //tFn.setCached(true);

    aStEditMode = nFn.create("stEditMode", "stEditMode", MFnNumericData::kBoolean, 0);
    nFn.setChannelBox(1);
    aStUiData = tFn.create("stUiData", "stUiData", MFnData::kString);

    //addAttribute(aStStartUp);

    std::vector<MObject> driverObjs = {
        aStEditMode, aStUiData,
        aStStartUp,
        aStStartMatrix, 
        aStStartTangent, aStStartTangentScale, aStStartAutoTangentBlend,
        aStEndMatrix, aStEndUp, aStEndTangent, aStEndTangentScale, aStEndAutoTangentBlend,
        aStNSpans, aStDegree
    };

    /*std::vector<MObject*>drivenObjs = {
        &aBalanceWheel,
        &aStOutCurve, &aStOutUpCurve
    };*/

    std::vector<MObject>drivenObjs = {
        aBalanceWheel,
        aStOutCurve,
        aStOutUpCurve
    };

    addAttributes<StrataCurve>(driverObjs);
    addAttributes<StrataCurve>(drivenObjs);
    // 
    
    

    //addAttribute(aStOutCurve); // vector attrs don't like nurbs curves
    setAttributesAffect<StrataCurve>(driverObjs, drivenObjs); // this crashes on the vector attributes

    //MCHECK(attributeAffects(aStStartUp, aBalanceWheel), "vector affects bw error");
    // 
    //MCHECK(attributeAffects(aStStartUp, aStOutCurve), "vector affects curve error"); //this one
    // 
    //attributeAffects(aStStartUp, aStOutUpCurve);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataCurve::compute(const MPlug& plug, MDataBlock& data) {
    MStatus s = MS::kSuccess;

    if (data.isClean(plug)) {
        return s;
    }
    //DEBUGS("curve compute");

    // build a basic curve
    MMatrix startMat = data.inputValue(aStStartMatrix).asMatrix();

    MVector startTan = data.inputValue(aStStartTangent).asVector() *data.inputValue(aStStartTangentScale).asFloat();

    //DEBUGS("start tan");
    //DEBUGS(startTan.x << startTan.y << startTan.z);

    //startTan = startMat * startTan;

    //DEBUGS("start tan mat mult");
    //DEBUGS(startTan.x << startTan.y << startTan.z);

    MMatrix endMat = data.inputValue(aStEndMatrix).asMatrix();
    MVector endTan = data.inputValue(aStEndTangent).asVector() *data.inputValue(aStEndTangentScale).asFloat();

    //data.setClean(plug);
    //return s;

    MFnNurbsCurve nurbsFn;
    MPointArray pointArr(4, MPoint());
    //MPointArray pointArr(
    //    {const MPoint(startMat.matrix[3]), const MPoint(startMat * startTan) },
    //    4
    //);


    pointArr[1] = MPoint(startTan) * startMat;
    pointArr[2] = MPoint(endTan) * endMat;
    pointArr[0] = MPoint(startMat.matrix[3]);
    pointArr[3] = MPoint(endMat.matrix[3]);

    // prevent the degree going beyond the number of spans in the curve
    uint degree = std::min(uint(data.inputValue(aStDegree).asInt()), pointArr.length() - 1);

    /*data.setClean(plug);
    return s;*/
    MFnNurbsCurveData dataFn;
    MObject dataParentObj = dataFn.create();
    /*MObject crvObj = nurbsFn.createWithEditPoints(pointArr, degree, MFnNurbsCurve::kOpen,
        false, true, true, dataParentObj);*/

    const MDoubleArray knotArr = uniformKnotsForCVs(pointArr.length(), degree);

    //DEBUGS("knot degree, vector:");
    //DEBUGS(degree);
    //DEBUGMVI(knotArr);

    MObject crvObj = nurbsFn.create(
        pointArr,
        knotArr,
        degree, 
        MFnNurbsCurve::kOpen,
        false,  
        false, 
        dataParentObj);

    //data.setClean(plug);
    //return s;
    data.outputValue(aStOutCurve).set(dataParentObj);

    //data.setClean(plug);
    //return s;

    data.setClean(aStOutCurve);  // crashes?

    // flip balancewheel
    data.outputValue(aBalanceWheel).setBool(
        !data.outputValue(aBalanceWheel).asBool());
    data.setClean(aBalanceWheel);

    data.setClean(plug);
    return s;
}
