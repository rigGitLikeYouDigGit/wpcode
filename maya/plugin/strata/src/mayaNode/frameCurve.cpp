

#include "frameCurve.h"
#include "../logger.h"

using namespace strata;
using namespace Eigen;

/*
TODO:
proper way of interpolating general influence on curves, then
eval-ing that to blend arbitrary value data (like upvector influence)

curve offsets, bind curves using ref curve and transform to active curve and upvectors

*/

DEFINE_STATIC_NODE_CPP_MEMBERS(FRAMECURVE_STATIC_MEMBERS, FrameCurveNode);

MTypeId FrameCurveNode::kNODE_ID(0x00122EA2);
MString FrameCurveNode::kNODE_NAME("frameCurve");


MStatus FrameCurveNode::initialize() {
    DEBUGS("Element initialize")
        MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;


    aCurveIn = tFn.create("curveIn", "curveIn", MFnData::kNurbsCurve);
    aRefCurveIn = tFn.create("refCurveIn", "refCurveIn", MFnData::kNurbsCurve);
    aSteps = nFn.create("steps", "steps", MFnNumericData::kInt, 20);
    nFn.setMin(1);


    // samples
    aSampleIn = cFn.create("sampleIn", "sampleIn");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);

    aRefSampleMatrixIn = mFn.create("refSampleMatrixIn", "refSampleMatrixIn");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aRefSampleMatrixIn);

    aActiveSampleMatrixIn = mFn.create("activeSampleMatrixIn", "activeSampleMatrixIn");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aActiveSampleMatrixIn);

    aSampleUIn = nFn.create("sampleUIn", "sampleUIn", MFnNumericData::kDouble, -0.01);
    nFn.setMin(-0.01);
    nFn.setMax(1.0);
    cFn.addChild(aSampleUIn);

    // upvectors
    aRefUp = cFn.create("refUp", "refUp");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);
    
    aRefUpMatrix = mFn.create("refUpMatrix", "refUpMatrix");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aRefUpMatrix);

    aActiveUp = cFn.create("activeUp", "activeUp");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);

    aActiveUpMatrix = mFn.create("activeUpMatrix", "activeUpMatrix");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aActiveUpMatrix);

    /// outputs
    aSampleOut = cFn.create("sampleOut", "sampleOut");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setWritable(false);

    aSampleMatrixOut = mFn.create("sampleMatrixOut", "sampleMatrixOut");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aSampleMatrixOut);

    aSampleMatrixOnCurveOut = mFn.create("sampleMatrixOnCurveOut", "sampleMatrixOnCurveOut");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aSampleMatrixOnCurveOut);

    aSampleUOut = nFn.create("sampleUOut", "sampleUOut", MFnNumericData::kDouble, 0.0);
    
    cFn.addChild(aSampleUOut);
   


    std::vector<MObject> drivers{
        aCurveIn,
        aRefCurveIn,
        aSteps,

        aRefSampleMatrixIn,
        aSampleUIn,

        aRefUpMatrix,
        aActiveUpMatrix
    };
    std::vector<MObject> driven{
        
        aSampleMatrixOut,
        aSampleMatrixOnCurveOut,
        aSampleUOut
    };

    std::vector<MObject> toAdd{
        aCurveIn,
        aRefCurveIn,
        aSteps,
        aSampleIn,
        aSampleOut,
        aRefUp,
        aActiveUp
    };



    addAttributes<FrameCurveNode>(toAdd);
    setAttributesAffect<FrameCurveNode>(drivers, driven);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    //DEBUGS("end element initialize")
    return s;
}

struct CurveSample {
    float u = 0;
    //Affine3f offset = Affine3f::Identity();
    MMatrix offset = MMatrix::identity;

};

inline void setMMatTranslation(MMatrix& mat, MVector v) {
    mat[3][0] = v[0];
    mat[3][1] = v[1];
    mat[3][2] = v[2];
}

inline MMatrix buildMMatrix(MVector x, MVector y, MVector z, MPoint pos) {
    MMatrix mat = MMatrix::identity;
    mat[0][0] = x[0];
    mat[0][1] = x[1];
    mat[0][2] = x[2];

    mat[1][0] = y[0];
    mat[1][1] = y[1];
    mat[1][2] = y[2];

    mat[2][0] = z[0];
    mat[2][1] = z[1];
    mat[2][2] = z[2];

    mat[3][0] = pos[0];
    mat[3][1] = pos[1];
    mat[3][2] = pos[2];
    return mat;
}

/* compute frames on ref curve - 
register samples against it

compute frames on active curve,
project out samples to it

TODO:
there's no point in Eigen here, I was just reusing
stuff from Strata, but it's very wasteful here
*/

void interpolateMMatrixArray2(std::vector<MMatrix>& mmatrixArr, MMatrix& out, float t) {
    /* assuming steadily spaced keypoints in arr, interpolate at param t
    slerp rotation component
    */
    t = fmin(fmax(t, 0.001f), 0.999f);

    int start = static_cast<int>(floor(float(mmatrixArr.size()) * t));
    int end = std::min(start + 1, int(mmatrixArr.size() - 1));
    float fraction = mmatrixArr.size() * t - start;

    MMatrix matA = mmatrixArr[start];
    MMatrix matB = mmatrixArr[end];

    MVector tanA(matA[0]);
    MVector tanB(matB[0]);


    MQuaternion qA(MVector::xAxis, tanA);
    MQuaternion qB(MVector::xAxis, tanB);
    MQuaternion qTan = slerp(qA, qB, fraction);
    
    MQuaternion qNA(MVector::zAxis, MVector(matA[2]));
    MQuaternion qNB(MVector::zAxis, MVector(matB[2]));
    MQuaternion qNorm = slerp(qNA, qNB, fraction);

    out = qTan * qNorm;

    MPoint posA(matA[3]);
    MPoint posB(matB[3]);

    MPoint posOut = posA + (posB - posA) * fraction;

    setMMatTranslation(out, MVector(posOut));

    return;
}

MVectorArray makeRMFNormals(
    MVectorArray& positions,
    MVectorArray& tangents,
    const MVectorArray& targetNormals,
    const int nSamples
) {/*
    as above, but working on only positions and tangents
    */

    MVectorArray resultNs;
    resultNs.setLength(nSamples);

    //Eigen::Vector3f ri = targetNormals.row(0);
    resultNs[0] = targetNormals[0];

    for (int i = 0; i < nSamples - 1; i++) {
        MVector xi = positions[i];
        MVector ti = tangents[i].normal();

        MVector xiPlus1 = positions[i + 1];
        MVector v1 = xiPlus1 - xi;
        double c1 = v1*  v1;
        double ttf = v1 * (resultNs[i]);
        MVector ttv = v1 * (2.0 / c1) * ttf;
        //MVector ttr = MVector(resultNs.row(i)) - ttv;
        MVector ttr = resultNs[i] - ttv;
        //Vector3f rLi = resultNs.row(i) - v1 * (2.0 / c1) * (v1.dot(resultNs.row(i)));
        MVector rLi = ttr;
        MVector tLi = ti - (2.0 / c1) * (v1 * ti) * v1;

        MVector tiPlus1 = tangents[i + 1].normal(); // next point's tangent
        MVector v2 = tiPlus1 - tLi;
        double c2 = v2 * v2 ;
        MVector riPlus1 = rLi - (2.0 / c2) * (v2 * (rLi)) * v2; // final reflected normal
        resultNs[i + 1] = riPlus1.normal();
    }
    return resultNs;
}




MStatus FrameCurveNode::compute(const MPlug& plug, MDataBlock& data) {

    LOG("frameCurves COMPUTE: " + MFnDependencyNode(thisMObject()).name() + " plug:" + plug.name());
    //l("isClean? " + str(data.isClean(plug)));
    MS s(MS::kSuccess);
    l.hush = 1;
    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }
    /* for now, require ref and active curve connection
    */
    MFnNurbsCurve activeFn(data.inputValue(aCurveIn).asNurbsCurve());
    l("activeFn point count: " + str(activeFn.numCVs()));
    
    MFnNurbsCurve refFn(data.inputValue(aRefCurveIn).asNurbsCurve());
    l("refFn point count: " + str(refFn.numCVs()));


    
    /* pull in point and tangent arrays from both curves for RMF*/
    int nSteps = data.inputValue(aSteps).asInt(); // TODO: make this adaptive somehow
    //MatrixX3f refPosMat(nSteps, 3);
    //MatrixX3f refTanMat(nSteps, 3);
    MVectorArray refPosArr;
    refPosArr.setLength(nSteps);
    MVectorArray refTanArr;
    refTanArr.setLength(nSteps);

    /// REF rmf
    for (int i = 0; i < nSteps; i++) { // TODO: parallel ref, active
        double u = (1.0 / float(nSteps -1)) * float(i);
        double domainMin;
        double domainMax;
        refFn.getKnotDomain(domainMin, domainMax);
        u = u * (domainMax - domainMin);
        MPoint pos;
        refFn.getPointAtParam(u, pos, MSpace::kWorld);
       
        //refPosMat.row(i) = toEigen<float>(pos);
        refPosArr[i] = MVector(pos);
        MVector tan = refFn.tangent(u, MSpace::kWorld).normal();
        //refTanMat.row(i) = toEigen<float>(tan);
        refTanArr[i] = tan;
        
        //l("passed ref bp");
    }

    l("ref arrs:");
    DEBUGAV(refTanArr);
    DEBUGAV(refPosArr);


    // for now only take +z from up matrix
    //MatrixX3f targetRefNormals(1, 3);
    MVectorArray targetRefNormals;
    targetRefNormals.setLength(1); // temp
    l("targetNormals made");
    //l(str(int(targetRefNormals.rows())));
    MArrayDataHandle refUpArrDH = data.inputArrayValue(aRefUp);
    std::vector<MMatrix> refMats(nSteps);

    if (refUpArrDH.elementCount()) {
        jumpToPhysicalIndex(refUpArrDH, 0);
        MMatrix upMat = refUpArrDH.inputValue().child(aRefUpMatrix).asMatrix();
        setMMatTranslation(upMat, MVector(0, 0, 0));
        MVector upVec = upMat * MVector(0, 0, 1);
        //targetRefNormals.row(0) = toEigen<float>(upVec);
        targetRefNormals[0] = upVec;
        l("set target Ns from matrix:");
        DEBUGMV(upVec);
    }
    else {
        //targetRefNormals.row(0) = Vector3f(0, 0, 1);
        targetRefNormals[0] = MVector(0, 0, 1);
    }
    l("targetNormals set");
    
    //MatrixX3f refNormals = makeRMFNormals(refPosMat, refTanMat,
    //    targetRefNormals, nSteps
    //);

    MVectorArray refNormals = makeRMFNormals(refPosArr, refTanArr,
        targetRefNormals, nSteps
    );
    DEBUGAV(refNormals);

    for (int i = 0; i < nSteps; i++) {
        //MVector bitangent(refTanMat.row(i).cross(refNormals.row(i)).data());
        MVector bitangent( refNormals[i] ^ refTanArr[i]);
        refMats[i] = buildMMatrix(
            refTanArr[i],
            bitangent,
            refNormals[i],
            refPosArr[i]
        );
    }
    l("built mmatrix vector");


    l("beginACTIVE");

    // ACTIVE rmf
        /* pull in point and tangent arrays from both curves for RMF*/
    //MatrixX3f activePosMat(nSteps, 3);
    //MatrixX3f activeTanMat(nSteps, 3);
    MVectorArray activePosArr;
    activePosArr.setLength(nSteps);
    MVectorArray activeTanArr;
    activeTanArr.setLength(nSteps);
    std::vector<MMatrix> activeMats(nSteps);
    

    for (int i = 0; i < nSteps; i++) { // TODO: parallel ref, active
        double u = (1.0 / float(nSteps - 1)) * float(i);
        double domainMin;
        double domainMax;
        activeFn.getKnotDomain(domainMin, domainMax);
        u = u * (domainMax - domainMin);
        MPoint pos;
        //activeFn.getPointAtParam(u, pos);
        //activePosMat.row(i) = toEigen<float>(pos);
        //MVector tan = activeFn.tangent(u);
        //activeTanMat.row(i) = toEigen<float>(tan);
        activeFn.getPointAtParam(u, pos, MSpace::kWorld);
        activePosArr[i] = MVector(pos);
        activeTanArr[i] = activeFn.tangent(u, MSpace::kWorld).normal();
    }

    // for now only take +z from up matrix
    //MatrixX3f targetActiveNormals(1, 3);
    MVectorArray targetActiveNormals;
    targetActiveNormals.setLength(1); // temp
    l("targetNormals made");
    //l(str(int(targetActiveNormals.rows())));
    MArrayDataHandle activeUpArrDH = data.inputArrayValue(aActiveUp);
    if (activeUpArrDH.elementCount()) {
        jumpToPhysicalIndex(activeUpArrDH, 0);
        MMatrix upMat = activeUpArrDH.inputValue().child(aActiveUpMatrix).asMatrix();
        setMMatTranslation(upMat, MVector());
        MVector upVec = upMat * MVector(0, 0, 1);
        targetActiveNormals[0] = upVec;
        l("set target Ns from matrix:");
        DEBUGMV(upVec); 
    }
    else {
        targetActiveNormals[0] = MVector(0, 0, 1);
    }
    l("targetNormals set");
    MVectorArray activeNormals = makeRMFNormals(activePosArr, activeTanArr,
        targetActiveNormals, nSteps
    );
    l("found active arrs:");
    DEBUGAV(activeTanArr);
    DEBUGAV(activeNormals);
    DEBUGAV(activePosArr);

    for (int i = 0; i < nSteps; i++) {
        MVector bitangent = activeNormals[i] ^ activeTanArr[i];
        activeMats[i] = buildMMatrix(
            activeTanArr[i],
            bitangent,
            activeNormals[i],
            activePosArr[i]
        );
    }

    /* pull in sample targets
    
    and also just multiply them out? don't need to do anything else here
    */
    MArrayDataHandle sampleInArrDH = data.inputArrayValue(aSampleIn);
    std::vector<CurveSample> sampleVec(sampleInArrDH.elementCount());
    MArrayDataHandle sampleOutArrDH = data.outputArrayValue(aSampleOut);
    int nSamples = sampleInArrDH.elementCount();
    for (int i = 0; i < nSamples; i++) {
        jumpToPhysicalIndex(sampleInArrDH, i);

        // if u < 0 , use matrix as nearest-point check on curve
        // else use it as a full offset from that point on ref curve
        MDataHandle sampleMatDH = sampleInArrDH.inputValue().child(aRefSampleMatrixIn);
        MDataHandle sampleUDH = sampleInArrDH.inputValue().child(aSampleUIn);
        
        MMatrix sampleRefMat = sampleMatDH.asMatrix();
        double u = sampleUDH.asDouble();
        if (u < 0.0) {
            MPoint atPos(sampleRefMat[3]);
            l("getting point at pos:");
            DEBUGMV(atPos);
            //s = refFn.getParamAtPoint(atPos, u, MSpace::kWorld);
            double uTarget = 0.5;
            atPos = refFn.closestPoint(atPos, &uTarget, 0.05, MSpace::kWorld);
            double domainMin;
            double domainMax;
            refFn.getKnotDomain(domainMin, domainMax);
            u = uTarget / (domainMax - domainMin);
            //MCHECK(s, "could not get param at point");
            if (s.error()) {
                l("STATUS ERROR getting param ");
            }
            l("found u:" + str(u));
        }
        l("interpolating at u:" + str(u));
        MMatrix refCurveMat;
        interpolateMMatrixArray2(
            refMats,
            refCurveMat,
            static_cast<float>(u)
        );

        l("ref mat pos:");
        DEBUGMV(MPoint(refCurveMat[3]));
        
        
        MMatrix offset = refCurveMat.inverse() * sampleRefMat;


        // multiply out to active curve
        jumpToPhysicalIndex(sampleOutArrDH, i);
        MDataHandle outElDH = sampleOutArrDH.outputValue();
        
        MMatrix activeCurveMat;
        interpolateMMatrixArray2(
            activeMats,
            activeCurveMat,
            static_cast<float>(u)
        );
         
        l("active mat pos:");
        DEBUGMV(MPoint(activeCurveMat[3]));

        MMatrix outMat = activeCurveMat * offset;
        outElDH.child(aSampleMatrixOnCurveOut).setMMatrix(activeCurveMat);
        outElDH.child(aSampleMatrixOut).setMMatrix(outMat);

        outElDH.child(aSampleUOut).setDouble(u);

    }

    data.setClean(aSampleMatrixOut);
        data.setClean(aSampleMatrixOnCurveOut);
        data.setClean(aSampleUOut);
    return MS::kSuccess;
}