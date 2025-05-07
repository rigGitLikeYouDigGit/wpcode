
#pragma once
#include <vector>
//#include <array>

#include "../macro.h"
#include "../api.h"
#include "../MInclude.h"

#include "../lib.h"
#include "../libEigen.h"

#include "matrixCurve.h"

using namespace ed;

MTypeId MatrixCurveNode::kNODE_ID(0x00122CA4);
MString MatrixCurveNode::kNODE_NAME("matrixCurve");

//MString MatrixCurveNode::drawDbClassification("drawdb/geometry/strataShape");
//MString MatrixCurveNode::drawRegistrantId("StrataShape");

//DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, MatrixCurveNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(MATCURVE_NODE_STATIC_MEMBERS, MatrixCurveNode)


MStatus MatrixCurveNode::initialize() {
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

    aCurveRootResIn = nFn.create("curveRootResIn", "curveRootIn", MFnNumericData::kInt, 5);
    nFn.setMin(1);

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
        aMatrixMidInMatrix,
        aCurveRootResIn,
    };
    std::vector<MObject> driven{
        aCurveOut,
        aSampleOutMatrix
    };

    std::vector<MObject> toAdd{
        aMatrixStartIn, 
        aMatrixEndIn,
        aMatrixMidIn,
        aCurveRootResIn,
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


MStatus MatrixCurveNode::updateMatrixCache(MDataBlock& data) {
    /* update the cached mmatrixArray on node 
    to recalculate output curve shape*/
    MS s(MS::kSuccess);


    MArrayDataHandle arrDH = data.inputArrayValue(aMatrixMidIn, &s);
    MCHECK(s, "ERROR getting array data handle ");
    int nSpans = 1 + arrDH.elementCount(&s);
    MCHECK(s, "ERROR getting arrayDH element count");
    MDataHandle rootResDH = data.inputValue(aCurveRootResIn, &s);
    MCHECK(s, "ERROR getting rootResDH");
    int rootRes = rootResDH.asInt();
    int nResultMats = (nSpans) * 
         rootRes + 1;

    std::vector<MMatrix> controlMats(nSpans + 1);
    //controlMats.resize(nSpans + 1);
    controlMats[0] = data.inputValue(aMatrixStartIn).asMatrix();
    for (int i = 0; i < nSpans - 1; i++) {
        jumpToElement(arrDH, i);
        controlMats[i + 1] = arrDH.inputValue().child(aMatrixMidInMatrix).asMatrix();
    }
    controlMats[nSpans] = data.inputValue(aMatrixEndIn).asMatrix();

    cachedMats = curveMatricesFromDriverDatas(
        controlMats, data.inputValue(aCurveRootResIn).asInt());

    return s;

}

MStatus MatrixCurveNode::updateCurve(MDataBlock& data) {
    /* create new nurbsCurve data object and set it as data output
    */
    MS s(MS::kSuccess);
    MPointArray curvePts(static_cast<int>(cachedMats.size()));
    for (unsigned int i = 0; i < curvePts.length(); i++) {
        curvePts[i] = MPoint(cachedMats[i].matrix[3]);
    }

    MFnNurbsCurveData dataFn(data.outputValue(aCurveOut).asNurbsCurve());

    MFnNurbsCurve::Form curveForm = MFnNurbsCurve::kOpen;
    if (cachedMats[0].isEquivalent(cachedMats.back())) {
        curveForm = MFnNurbsCurve::kClosed;
    }

    MObject newCurveObj = MFnNurbsCurve().createWithEditPoints(
        curvePts, 2, curveForm,
        false, true, true,
        dataFn.object()
    );
    data.outputValue(aCurveOut).setMObject(dataFn.object());
    data.setClean(aCurveOut);
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

    std::vector<MObject> curveAffectors{
        aMatrixStartIn, aMatrixMidInMatrix,
        aMatrixEndIn, aCurveRootResIn
    };

    if (!attrsClean(curveAffectors, data) )
    {
        s = updateMatrixCache(data);
        MCHECK(s, "ERROR updating matrix cache");
        s = updateCurve(data);
        MCHECK(s, "ERROR updating output curve shape");

        setAttrsClean(curveAffectors, data);
        data.setClean(aCurveOut);
    }

    data.setClean(plug);

    return s;
}





