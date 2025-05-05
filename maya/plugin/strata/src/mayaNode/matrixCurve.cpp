
#pragma once
#include <vector>
//#include <array>

#include "../macro.h"
#include "../api.h"
#include "../MInclude.h"

#include "matrixCurve.h"

using namespace ed;

MTypeId MatrixCurveNode::kNODE_ID(0x00122CA4);
MString MatrixCurveNode::kNODE_NAME("matrixCurve");

//MString MatrixCurveNode::drawDbClassification("drawdb/geometry/strataShape");
//MString MatrixCurveNode::drawRegistrantId("StrataShape");

//DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, MatrixCurveNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(MATCURVE_NODE_STATIC_MEMBERS, MatrixCurveNode)


MStatus MatrixCurveNode::initialize() {
    DEBUGSL("shape initialize")
        MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    aMatrixStartIn = mFn.create("matrixStartIn", "matrixStartIn");
    mFn.setDefault(MMatrix::identity);
    mFn.setReadable(false);
    aMatrixEndIn = mFn.create("matrixEndIn", "matrixEndIn");
    mFn.setDefault(MMatrix::identity);
    mFn.setReadable(false);

    aMatrixMidIn = cFn.create("matrixMidIn", "matrixMidIn");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);
    // call me byron
    aMatrixMidInMatrix = mFn.create("matrixMidInMatrix", "matrixMidInMatrix");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aMatrixMidInMatrix);

    aSampleIn = cFn.create("sampleIn", "sampleIn");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);
    aSampleInParam = nFn.create("sampleInParam", "sampleInParam", MFnNumericData::kFloat, 0.0);
    nFn.setMin(0.0);
    nFn.setMax(1.0);
    cFn.addChild(aSampleInParam);

    aSampleOut = cFn.create("sampleOut", "sampleOut");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setWritable(false);
    aSampleOutMatrix = mFn.create("sampleOutMatrix", "sampleOutMatrix");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aSampleOutMatrix);

    aCurveOut = tFn.create("curveOut", "curveOut", MFnData::kNurbsCurve);
    tFn.setDefault(MFnNurbsCurveData().create());
    tFn.setWritable(false);

    /* actually not using the brute-force NxN attributeAffects,
    since calculating the curve is quite expensive*/
    std::vector<MObject> drivers{
        aMatrixStartIn,
        aMatrixEndIn,
        aMatrixMidInMatrix
    };
    std::vector<MObject> driven{
        aCurveOut,
        aSampleOutMatrix
    };

    std::vector<MObject> toAdd{
        aMatrixStartIn, 
        aMatrixEndIn,
        aMatrixMidIn,
        aSampleIn,
        aSampleOut,
        aCurveOut,
    };

    addAttributes<thisT>(toAdd);
    setAttributesAffect<thisT>(drivers, driven);
    attributeAffects(aSampleInParam, aSampleOut);
    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}

MStatus MatrixCurveNode::compute(const MPlug& plug, MDataBlock& data) {
    /* 
    */
    MS s(MS::kSuccess);
    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }



    data.setClean(plug);

    return s;
}





