
/*
* some portions below are taken from the Free Electron project - 
* at time of writing they are distributed under the BSD 2-clause license,
* with the following notice:
Copyright(C) 2003 - 2021 Free Electron Organization
Any use of this software requires a license.If a valid license
was not distributed with this file, visit freeelectron.org.* /

*/

#pragma once
#include <vector>
//#include <array>

#include "../macro.h"
#include "../api.h"
#include "../MInclude.h"
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/Splines>


#include "../lib.h"
#include "../libEigen.h"

#include "matrixCurve.h"
#include "../libnurbs.h"
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

    aMatrixRootIterationsIn = nFn.create("matrixRootIterationsIn", "matrixRootIterationsIn",
        MFnNumericData::kInt, 1);
    nFn.setChannelBox(true);
    nFn.setReadable(false);


    aCurveRootResIn = nFn.create("curveRootResIn", "curveRootIn", MFnNumericData::kInt, 5);
    nFn.setMin(1);
    nFn.setKeyable(true);
    nFn.setChannelBox(true);

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
        aMatrixRootIterationsIn,
    };
    std::vector<MObject> driven{
        aCurveOut,
        aSampleOutMatrix
    };

    std::vector<MObject> toAdd{
        aMatrixStartIn, 
        aMatrixEndIn,
        aMatrixMidIn,
        aMatrixRootIterationsIn,
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

//* static
template<typename T>
void interpolateTransform(
    Eigen::Matrix4<T>& a_result,
    const Eigen::Matrix4<T>& a_matrix1, 
    const Eigen::Matrix4<T>& a_matrix2,
    T a_fraction, 
    bool a_bezier)
{
    //	feLog("CurveCreateOp::interpolateTransform fraction %.6G\n"
    //			"from\n%s\nto\n%s\n",
    //			a_fraction,c_print(a_matrix1),c_print(a_matrix2));
    
    Eigen::Matrix4<T> inv1 = a_matrix1.inverse();
    //invert(inv1, a_matrix1);

    //const SpatialTransform delta12 = a_matrix2 * inv1;
    const Eigen::Matrix4<T> delta12 = a_matrix2 * inv1;

    //SpatialTransform partial12;
    Eigen::Matrix4<T> partial12;

    if (a_bezier)
    {
        //MatrixBezier<SpatialTransform> matrixBezier;
        MatrixBezier<Eigen::Matrix4<T>> matrixBezier;
        matrixBezier.solve(partial12, delta12, a_fraction);
    }
    else
    {
        Eigen::MatrixPower<Eigen::Matrix4f> relMatPower(delta12);
        partial = relMatPower(a_fraction);
        
    }

    a_result = partial12 * a_matrix1;
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

    fullSpanMatPowers.clear();
    halfSpanMatPowers.clear();




    //std::vector<MMatrix> controlMats(nSpans + 1);
    MMatrixArray controlMats(nSpans + 1);
    MVectorArray spanVecs(nSpans);
    //controlMats.resize(nSpans + 1);
    controlMats[0] = data.inputValue(aMatrixStartIn).asMatrix();
    for (int i = 0; i < nSpans - 1; i++) {
        jumpToElement(arrDH, i);
        controlMats[i + 1] = arrDH.inputValue().child(aMatrixMidInMatrix).asMatrix();
    }
    controlMats[nSpans] = data.inputValue(aMatrixEndIn).asMatrix();

    for (int i = 0; i < nSpans; i++) {
        spanVecs[i] = MPoint(controlMats[i + 1][3]) - MPoint(controlMats[i][3]);
    }

    // same matrix passed start and end
    bool closed = controlMats[0].isEquivalent(controlMats[controlMats.length() - 1]);

    cachedMats = controlMats;
    //return s;

    // orient control matrices properly
    MVectorArray tangents(controlMats.length());
    if (tangents.length() == 1) {
        // invalid for now
        return s;
    }
    if (tangents.length() == 2) {
        if (controlMats[0].isEquivalent(controlMats[1])) {
            return s;
        }
        tangents[0] = MPoint(controlMats[1][3]) - MPoint(controlMats[0][3]);
        tangents[0].normalize();
        tangents[1] = tangents[0];
        
    }

    else {
        //DEBUGSL("do tans:")
        for (int unsigned i = 0; i < tangents.length(); i++) {
            MVector tan(1, 0, 0);
            MPoint thisPoint(controlMats[i][3]);
            /*DEBUGS(std::to_string(i) + " start tan:");
            DEBUGMV(tan);*/
            int prevI = (i - 1) % tangents.length();
            int nextI = (i + 1) % tangents.length();
            MPoint nextPoint(controlMats[nextI][3]);
            MPoint prevPoint(controlMats[prevI][3]);

            // test taking tangent as vector from prev point to next
            // still need to scale them properly
            tan = nextPoint - prevPoint;


            //bool isEnd = ((i == 0) || (nextI == 0));
            //if ((!isEnd) || closed) {
            //    tan = (MVector(thisPoint - prevPoint) + MVector(nextPoint - thisPoint) + MVector(EPS, EPS, EPS)) / 2.0;
            //}
            //else {
            //    if (i == 0) {
            //        tan = cross(nextPoint - thisPoint;
            //    }
            //    if (nextI == 0) {
            //        tan = thisPoint - prevPoint;
            //    }
            //}
            tan.x = tan.x + EPS;
            tan.normalize();
            tangents[i] = tan;
        }

        if (!closed) {
            MVector scaledStartTan = tangents[1] * (sqrt(1.0 + spanVecs[0].length()) - 1.0);
            tangents[0] = (MVector(controlMats[1][3]) - scaledStartTan) - MVector(controlMats[0][3]);
            tangents[0].normalize();
            
            int lastI = tangents.length() - 1;
            MVector scaledEndTan = tangents[lastI - 1] * (sqrt(1.0 + spanVecs[lastI-1].length()) - 1.0);
            tangents[lastI] = -((MVector(controlMats[lastI-1][3]) + scaledEndTan) - MVector(controlMats[lastI][3]));
            tangents[lastI].normalize();
        }
    }
    DEBUGSL("TANS:");
    
    //return s;
    // set new matrices from tangents
    for (unsigned int i = 0; i < controlMats.length(); i++) {
        
        DEBUGMV(tangents[i]);

        setRow(controlMats[i], 0, tangents[i]);
        //setRow(controlMats[i], 1, (tangents[i] ^ MVector(controlMats[i][2])));
        setRow(controlMats[i], 1, -(tangents[i] ^ MVector(getRow(controlMats[i], 2))));
        //setRow(controlMats[i], 0, MVector(controlMats[i][1]) ^ MVector(controlMats[i][2]));
        //setRow(controlMats[i], 2, MVector(controlMats[i][0]) ^ MVector(controlMats[i][1]));
        setRow(controlMats[i], 2, MVector(getRow(controlMats[i], 0)) ^ MVector(getRow(controlMats[i], 1)));
    }
    //return s;
    cachedMats = controlMats;
    //return s;

    cachedMats = curveMatricesFromDriverDatas(
        controlMats, data.inputValue(aCurveRootResIn).asInt(),
        data.inputValue(aMatrixRootIterationsIn).asInt()
        );

    

    return s;

}

MStatus MatrixCurveNode::updateCurve(MDataBlock& data) {
    /* create new nurbsCurve data object and set it as data output
    */
    MS s(MS::kSuccess);
    int nMats = static_cast<int>(cachedMats.length());
    MPointArray curvePts(nMats);
    MVectorArray curvePtVecs(static_cast<int>(cachedMats.length()));
    //Eigen::ArrayX3d eiPointArr(static_cast<int>(cachedMats.length()));
    //Eigen::ArrayXXd eiPointArr(static_cast<int>(cachedMats.length()), 3);
    //Eigen::MatrixXd eiPointArr(nMats, 3);
    Eigen::MatrixXd eiPointArr(3, nMats);
    //std::vector<Eigen::Vector3f> eiPointArr(nMats);

    Eigen::MatrixXd eiTanArr(3, nMats);

    for (unsigned int i = 0; i < curvePts.length(); i++) {
        auto d = cachedMats[i].matrix[3];
        curvePts[i] = MPoint(d);
        //curvePtVecs[i] = MVector(MPoint(cachedMats[i].matrix[3]));
        /*eiPointArr(i, 0) = d[0];
        eiPointArr(i, 1) = d[1];
        eiPointArr(i, 2) = d[2];*/
        /*eiPointArr(0, i) = d[0];
        eiPointArr(0, i) = d[1];
        eiPointArr(0, i) = d[2];*/
        /*eiPointArr[i][0] = d[0];
        eiPointArr[i][1] = d[1];
        eiPointArr[i][2] = d[2];*/
        eiPointArr(0, i) = d[0];
        eiPointArr(1, i) = d[1];
        eiPointArr(2, i) = d[2];

        auto t = cachedMats[i].matrix[0];
        eiTanArr(0, i) = t[0];
        eiTanArr(1, i) = t[1];
        eiTanArr(2, i) = t[2];

    }


    //MVectorArray curvePts(static_cast<int>(cachedMats.length()));
    //for (unsigned int i = 0; i < curvePts.length(); i++) {
    //    curvePts[i] = MPoint(cachedMats[i].matrix[3]);
    //}

    MFnNurbsCurveData dataFn(data.outputValue(aCurveOut).asNurbsCurve());

    MFnNurbsCurve::Form curveForm = MFnNurbsCurve::kOpen;
    if (cachedMats[0].isEquivalent(cachedMats[cachedMats.length()-1])) {
        curveForm = MFnNurbsCurve::kClosed;
    }
    int degree = 2;




    ////Eigen::Spline3d outSpline()
    ////Eigen::Spline<double, 3, 2> outSpline(knots, curvePts);


    Eigen::SplineFitting<Eigen::Spline3d> splineFit;

    //Eigen::Spline3d outSpline = splineFit.Interpolate(eiPointArr, degree);
    auto outSpline = splineFit.Interpolate(eiPointArr, degree);

    unsigned int denseRes = 20;
    curvePts.setLength(denseRes);

    //auto pt = outSpline(0.5);


    for (unsigned int i = 0; i < denseRes; i++) {
        double u = (1.0 / double(denseRes - 1)) * double(i);
        auto pt = outSpline(u);
        curvePts[i] = MPoint(pt[0], pt[1], pt[2]);
    }

    MDoubleArray knots = uniformKnotsForCVs(curvePts.length(), degree);


    /*MObject newCurveObj = MFnNurbsCurve().createWithEditPoints(
        curvePts, 2, curveForm,
        false, true, true,
        dataFn.object()
    );*/
    // maya's native edit points give crazy loops at sharp corners

    DEBUGSL("knots:");
    DEBUGMVI(knots);
    MObject newCurveObj = MFnNurbsCurve().create(
        curvePts,
        knots,
        degree, curveForm, false, true, dataFn.object()
    );
    if (dataFn.object().isNull()) {
        DEBUGSL("Data object is NULL");
    }
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





