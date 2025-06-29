
#pragma once
#include <vector>
//#include <array>

#include "../macro.h"
#include "../api.h"
#include "../MInclude.h"

#include "strataElementOpNode.h"
#include "strataGraphNode.h"
#include "strataPointNode.h"
#include "strataOpNodeBase.h"
#include "../lib.h"
#include "../libEigen.h"
#include "../libNurbs.h"
#include "../stringLib.h"
#include "../strataop/elementOp.h"

using namespace ed;

MTypeId StrataElementOpNode::kNODE_ID(0x00122CA2);
MString StrataElementOpNode::kNODE_NAME("strataElementOp");

MString StrataElementOpNode::drawDbClassification("drawdb/geometry/strataElementOp");
MString StrataElementOpNode::drawRegistrantId("StrataElementOpNodePlugin");

DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataElementOpNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(STRATAELEMENTOPNODE_STATIC_MEMBERS, StrataElementOpNode)
//MObject StrataElementOpNode::aStInput; MObject StrataElementOpNode::aStOpName; MObject StrataElementOpNode::aStOutput;

///MObject StrataElementOpNode::aStElement; MObject StrataElementOpNode::aStType; MObject StrataElementOpNode::aStName; MObject StrataElementOpNode::aStDriverExp; MObject StrataElementOpNode::aStGlobalIndex; MObject StrataElementOpNode::aStElTypeIndex; MObject StrataElementOpNode::aStFitTransform; MObject StrataElementOpNode::aStFitCurve;


MStatus StrataElementOpNode::initialize() {
    DEBUGS("Element initialize")
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;
    

    aStElement = cFn.create("stElement", "stElement");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    cFn.setReadable(false);
    cFn.setAffectsAppearance(true);

    /*expression to generate given elements - leave blank for raw points
    */
    aStDriverExp = tFn.create("stDriverExp", "stDriverExp", MFnData::kString); 
    tFn.setDefault(MFnStringData().create(""));

    /* expression for spaces of given elements*/
    aStSpaceExp = tFn.create("stSpaceExp", "stSpaceExp", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));

    /* name of new element to create*/
    aStName = tFn.create("stName", "stName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    

    cFn.addChild(aStDriverExp);
    cFn.addChild(aStSpaceExp);
    cFn.addChild(aStName);

    //// element attributes
    aStMatchWorldSpaceIn = nFn.create("stMatchWorldSpaceIn", "stMatchWorldSpaceIn", MFnNumericData::kFloat, 1.0);
    cFn.addChild(aStMatchWorldSpaceIn);
    aStDriverWeightIn = nFn.create("stDriverWeightIn", "stInDriverWeightIn", MFnNumericData::kFloat, 1.0);
    nFn.setArray(true);
    nFn.setUsesArrayDataBuilder(true);
    cFn.addChild(aStDriverWeightIn);

    // point attributes
    aStPointWorldMatrixIn = mFn.create("stPointWorldMatrixIn", "stPointWorldMatrixIn");
    mFn.setDefault(MMatrix());
    cFn.addChild(aStPointWorldMatrixIn);
    cFn.setAffectsAppearance(true);
    cFn.setAffectsWorldSpace(true);

    aStPointDriverLocalMatrixIn = mFn.create("stPointDriverLocalMatrixIn", "stPointDriverLocalMatrixIn");
    mFn.setDefault(MMatrix()); /* separate local matrix per driver*/
    mFn.setArray(true);
    mFn.setUsesArrayDataBuilder(true);
    cFn.addChild(aStPointDriverLocalMatrixIn);

    //// edge attributes
    aStEdgeCurveIn = tFn.create("stEdgeCurveIn", "stEdgeCurveIn", MFnData::kNurbsCurve);
    tFn.setDefault(MFnNurbsCurveData().create());
    cFn.addChild(aStEdgeCurveIn);


    ///// OUTPUTS
    aStElementOut = cFn.create("stElementOut", "stElementOut");
    cFn.setWritable(false);
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);

    aStNameOut = tFn.create("stNameOut", "stNameOut", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStNameOut);
    /* global index of new element created */
    aStGlobalIndex = nFn.create("stGlobalIndex", "stGlobalIndex", MFnNumericData::kInt, -1);
    nFn.setKeyable(false);
    nFn.setMin(-1);
    cFn.addChild(aStGlobalIndex);
    /* component-type-specific index of element created */
    aStElTypeIndex = nFn.create("stElTypeIndex", "stElTypeIndex", MFnNumericData::kInt, -1);
    nFn.setKeyable(false);
    nFn.setMin(-1);
    cFn.addChild(aStElTypeIndex);
    aStTypeOut = eFn.create("stTypeOut", "stTypeOut", 0);
    eFn.addField("point", 0);
    eFn.addField("edge", 1);
    eFn.addField("face", 2);
    cFn.addChild(aStTypeOut);

    aStPointFinalWorldMatrixOut = mFn.create("stPointFinalWorldMatrixOut", "stPointFinalWorldMatrixOut");
    mFn.setDefault(MMatrix());
    cFn.addChild(aStPointFinalWorldMatrixOut);

    aStPointWeightedDriverMatrixOut = mFn.create("stPointWeightedDriverMatrixOut", "stPointWeightedDriverMatrixOut");
    mFn.setDefault(MMatrix());
    cFn.addChild(aStPointWeightedDriverMatrixOut);
    aStPointWeightedLocalOffsetMatrixOut = mFn.create("stPointWeightedLocalOffsetMatrixOut", "stPointWeightedLocalOffsetMatrixOut");
    mFn.setDefault(MMatrix());
    mFn.setArray(true);
    mFn.setUsesArrayDataBuilder(true);
    cFn.addChild(aStPointWeightedLocalOffsetMatrixOut);
    aStPointDriverMatrixOut = mFn.create("stPointDriverMatrixOut", "stPointDriverMatrixOut");
    mFn.setDefault(MMatrix());
    mFn.setArray(true);
    mFn.setUsesArrayDataBuilder(true);
    cFn.addChild(aStPointDriverMatrixOut);




    aStEdgeCurveOut = tFn.create("stEdgeCurveOut", "stEdgeCurveOut", MFnData::kNurbsCurve);
    tFn.setDefault(MFnNurbsCurveData().create());
    cFn.addChild(aStEdgeCurveOut);

    //aStEdgeResolution = nFn.create("stEdgeResolution", "stEdgeResolution", MFnNumericData::kInt, 1);
    //nFn.setMin(1);
    //aStEdgeNormaliseParam = nFn.create("stEdgeNormaliseParam", "stEdgeNormaliseParam", MFnNumericData::kFloat, 1.0);
    //nFn.setMin(0.0);
    //nFn.setMin(1.0);
    //aStEdgeUseLength = nFn.create("stEdgeUseLength", "stEdgeUseLength", MFnNumericData::kFloat, 1.0);
    //nFn.setMin(0.0);
    //nFn.setMin(1.0);
    //aStEdgeReverse = nFn.create("stEdgeReverse", "stEdgeReverse", MFnNumericData::kFloat, 0.0);
    //nFn.setMin(0.0);
    //nFn.setMin(1.0);
    //// edge start/end entries - unsure if these should be all one array instead
    //aStEdgeStartIndex = nFn.create("stEdgeStartIndex", "stEdgeStartIndex", MFnNumericData::kInt, -1);
    //nFn.setMin(-1);
    //aStEdgeStartName = tFn.create("stEdgeStartName", "stEdgeStartName", MFnData::kString);
    //tFn.setDefault(MFnStringData().create(""));
    //
    //aStEdgeEndIndex = nFn.create("stEdgeEndIndex", "stEdgeEndIndex", MFnNumericData::kInt, -1);
    //nFn.setMin(-1);
    //aStEdgeEndName = tFn.create("stEdgeEndName", "stEdgeEndName", MFnData::kString);
    //tFn.setDefault(MFnStringData().create(""));

    //MFnCompoundAttribute edgeMidFn;
    //aStEdgeMid = edgeMidFn.create("stEdgeMid", "stEdgeMid");
    //edgeMidFn.setArray(true);
    //edgeMidFn.setUsesArrayDataBuilder(true);
    //aStEdgeMidIndex = nFn.create("stEdgeMidIndex", "stEdgeMidIndex", MFnNumericData::kInt, -1);
    //nFn.setMin(-1);
    //aStEdgeMidName = tFn.create("stEdgeMidName", "stEdgeMidName", MFnData::kString);
    //tFn.setDefault(MFnStringData().create(""));
    //edgeMidFn.addChild(aStEdgeMidIndex);
    //edgeMidFn.addChild(aStEdgeMidName);

    //// face attributes
    //MFnCompoundAttribute faceDriverFn;
    //aStFaceDriver = faceDriverFn.create("stFaceDriver", "stFaceDriver");
    //faceDriverFn.setArray(true);
    //faceDriverFn.setUsesArrayDataBuilder(true);
    //aStFaceDriverIndex = nFn.create("stFaceDriverIndex", "stFaceDriverIndex", MFnNumericData::kInt, -1);
    //nFn.setMin(-1);
    //aStFaceDriverName = tFn.create("stFaceDriverName", "stFaceDriverName", MFnData::kString);
    //faceDriverFn.addChild(aStFaceDriverIndex);
    //faceDriverFn.addChild(aStFaceDriverName);
    
    //addChildAttributes(cFn, 
    //    { &aStDriverExp, &aStType, &aStName, &aStGlobalIndex, &aStElTypeIndex,
    //    &aStPointInWorldMatrix, &aStPointOutFinalDriverMatrix, &aStPointOutFinalLocalOffsetMatrix, &aStPointOutFinalWorldMatrix,
    //    &aStEdgeResolution, &aStEdgeNormaliseParam, &aStEdgeUseLength}
    //);

    std::vector<MObject> drivers{
        //aStElement,
        aStName,
        aStDriverExp,
        aStSpaceExp,

        aStMatchWorldSpaceIn,
        aStDriverWeightIn,

        aStPointWorldMatrixIn,
        aStPointDriverLocalMatrixIn,
        //aStPointWeightedDriverMatrixOut,
        //aStPointWeightedLocalOffsetMatrixOut,
        //aStPointFinalWorldMatrixOut,

        aStEdgeCurveIn
    };
    std::vector<MObject> driven{
        //aStElementOut,
        aStNameOut,
        aStGlobalIndex,
        aStElTypeIndex,
        aStTypeOut,

        aStPointFinalWorldMatrixOut,

        aStEdgeCurveOut
    };

    std::vector<MObject> toAdd{
        aStElement,
        aStElementOut
    };

    s = addStrataAttrs<StrataElementOpNode>(drivers, driven, toAdd);
    MCHECK(s, "could not add Strata attrs");

    addAttributes<thisT>(toAdd);
    setAttributesAffect<thisT>(drivers, driven);

    attributeAffects(aStPointWorldMatrixIn, aStOutput);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    //DEBUGS("end element initialize")
    return s;
}

//MStatus StrataElementOpNode::setDependentsDirty(const MPlug& plugBeingDirtied,
//    MPlugArray& affectedPlugs
//) {
//    LOG("EL DEPENDENTS DIRTY: " + plugBeingDirtied.name());
//    MS s(MS::kSuccess);
//    
//    affectedPlugs.append(MPlug(thisMObject(), aStOutput));
//    //affectedPlugs.append(MPlug(thisMObject(), aStElement));
//    affectedPlugs.append(MPlug(thisMObject(), aStElementOut));
//
//    return s;
//}


void StrataElementOpNode::postConstructor() {
    LOG("element postConstructor");
    setExistWithoutInConnections(true);
    setExistWithoutOutConnections(true);
    /* cache starting name? */

    superT::postConstructor<thisT>(thisMObject());
}

MStatus StrataElementOpNode::legalConnection(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc,
    bool& isLegal
) const {
    DEBUGSL("element legalConnection")
    return superT::legalConnection<thisT>(
        plug,
        otherPlug,
        asSrc,
        isLegal
    );
}

MStatus StrataElementOpNode::connectionMade(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc
) {
    LOG("el connection made")
        MStatus s = superT::connectionMade<thisT>(
            thisMObject(),
            plug,
            otherPlug,
            asSrc
        );
    return MPxNode::connectionMade(
        plug, otherPlug, asSrc);
}

MStatus StrataElementOpNode::connectionBroken(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc
) {
    DEBUGSL("el connection broken")
        MStatus s = superT::connectionBroken<thisT>(
            thisMObject(),
            plug,
            otherPlug,
            asSrc
        );
    return MPxNode::connectionBroken(
        plug,
        otherPlug,
        asSrc
    );
}


MStatus StrataElementOpNode::edgeDataFromRawCurve(MStatus& ms, MObject& nodeObj, MDataBlock& data, MDataHandle& elDH, SEdgeData& eData) {
    /* if you specify only a raw curve without driver data (not recommended),
    build edge data from it - from maya this will be a nurbs curve
    
    sample the curve for each of its control points, and from that build raw spline for edge data
    */
    MDataHandle curveDH = elDH.child(aStEdgeCurveIn);
    MObject& curveObj = curveDH.asNurbsCurve();
    MFnNurbsCurveData dataFn(curveObj, &ms);
    MCHECK(ms, "Error retrieving nurbs curve data for plug");
    
    MFnNurbsCurve cFn(dataFn.object());

    if (cFn.numCVs() < 2) {
        DEBUGSL("Invalid curve passed for literal edge")
        return MS::kFailure;
    }

    // driver data for each cv, and for each half-span?
    eData.driverDatas.resize(cFn.numCVs() * 2 - 1);

    // set up point list for edge
    //Eigen::Matrix3Xd curvePoints(cFn.numCVs());

    // need to remap uniform param into knot space
    double uStart, uEnd;
    cFn.getKnotDomain(uStart, uEnd);

    for (int i = 0; i < cFn.numCVs() * 2 - 1; i++) {
        eData.driverDatas[i].index = -1;
        double u = 1.0 / (cFn.numCVs() - 1) * i;
        u = lerp(uStart + 0.0001, uEnd - 0.0001, u);
        MPoint pt;
        MVector tan;
        ms = cFn.getDerivativesAtParm(u, pt, tan, MSpace::kObject);
        MCHECK(ms, "invalid sample point on curve");

        Status s;
        makeFrame(s,
            eData.driverDatas[i].finalMatrix,
            //Eigen::Vector3f{ pt.x, pt.y, pt.z },
            toEigen<float>(pt),
            //Eigen::Vector3f{ tan.x, tan.y, tan.z }
            toEigen<float>(tan)
        );
    }
    return ms;
}

MStatus StrataElementOpNode::syncStrataParams(MObject& nodeObj, MDataBlock& data,
    StrataOp* thisOpPtr, StrataOpGraph* graphPtr) {
    /* build map of element names to expressions, from maya node
    */
    LOG("EL OP sync strata params");
    MS s;
    //thisStrataOpT* opPtr = getStrataOp<thisT>(nodeObj);
    thisStrataOpT* opPtr = static_cast<thisStrataOpT*>(thisOpPtr);
    if (opPtr == nullptr) {
        MGlobal::displayError(NODENAME + "COULD NOT RETRIEVE OP to sync");
        //NODELOG("COULD NOT RETRIEVE OP to sync");
        return MS::kFailure;
    }

    opPtr->paramMap.clear();
    opPtr->paramNames.clear();

    opPtr->opPointDataMap.clear();

    // check names currently found, remove any missing
    std::unordered_set<std::string> foundNames;

    MArrayDataHandle elDH = data.inputArrayValue(aStElement, &s);
    MCHECK(s, NODENAME + "error getting input array value");
    DEBUGSL("n array params to sync: " + std::to_string(elDH.elementCount()));

    MArrayDataHandle elOutDH = data.outputArrayValue(aStElementOut, &s);

    for (unsigned int i = 0; i < elDH.elementCount(); i++) {
        s = jumpToElement(elDH, i);
        MCHECK(s, "could not jump to element" + std::to_string(i));

        ElOpParam param;
        MDataHandle nameDH = elDH.inputValue().child(aStName);
        MDataHandle driverExpDH = elDH.inputValue().child(aStDriverExp);

        // check if name is empty
        std::string elName = nameDH.asString().asChar();
        trimEnds(elName);
        if (elName.empty()) { // empty name
            continue;
        }

        l("paramName: " + elName);

        param.name = elName;

        opPtr->paramMap.insert_or_assign(std::string(param.name), param); // crashes
        opPtr->paramNames.push_back(std::string(param.name));
        //// driver exp
        std::string driverExpStr = driverExpDH.asString().asChar();
        foundNames.insert(elName);
        param.driverExp.setSource(driverExpStr.c_str());
        //continue;
        // parent exp
        std::string spaceExpStr = elDH.inputValue().child(aStSpaceExp).asString().asChar();
        param.spaceExp.setSource(spaceExpStr.c_str());
        std::string elParentName = elName + "!";
        foundNames.insert(elParentName);


        // pull in element data
        float matchWorldSpace = elDH.inputValue().child(aStMatchWorldSpaceIn).asFloat();
        // always match in worldspace for now

        MMatrix mayaMat = elDH.inputValue().child(aStPointWorldMatrixIn).asMatrix();
        l("maya point mat: ");
        COUT << mayaMat << std::endl;
        l(" ");
        COUT << toEigen(mayaMat) << std::endl;
        param.pData.finalMatrix = toEigen(mayaMat);

        //opPtr->paramMap.insert({ param.name, param }); // crashes
        opPtr->paramMap.insert_or_assign( param.name, param ); // crashes
        //opPtr->paramMap.insert({ "test", param }); // no crash

    }

    for (auto& i : opPtr->paramMap) { // crashes on iteration
        if (foundNames.find(i.first) == foundNames.end()) {
            opPtr->paramMap.erase(i.first);
        }
    }
    return s;
}

MStatus StrataElementOpNode::compute(const MPlug& plug, MDataBlock& data) {

    LOG("EL COMPUTE: " + MFnDependencyNode(thisMObject()).name() + " plug:" + plug.name());
    l("isClean? " + str(data.isClean(plug)));
    MS s(MS::kSuccess);

    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }

    // pass to bases, compute strata op
    s = superT::compute<thisT>(thisMObject(), plug, data);
    //if (data.isClean(plug)) {
    //    l("plug marked clean, returning");
    //    return MS::kSuccess;
    //}
    if (s == MS::kEndOfFile) {
        return MS::kSuccess;
    }
    MCHECK(s, NODENAME + " ERROR in strata bases compute, halting");
    //DEBUGSL("strata base computed, back to elOp scope");

    l("el op comput complete, setting outputs");

    //return s;
    // update index attrs from op elements
    thisStrataOpT* opPtr = getStrataOp<thisT>(thisMObject()); /* this can be null?????*/
    if (opPtr == nullptr) {
        l("OP PTR IS NULL AFTER EL OP COMPUTE, halting");
        return MS::kFailure;
    }
    l("after compute op get size: " + std::to_string(opPtr->elementsAdded.size()));
    MArrayDataHandle elArrDH = data.outputArrayValue (aStElementOut);
    StrataManifold& manifold = opPtr->value();
    //for (unsigned int i = 0; i < elArrDH.elementCount(); i++){
    for (unsigned int i = 0; i < static_cast<unsigned int>(opPtr->elementsAdded.size()); i++){
        l("try jump to output element: " + std::to_string(i));
        //continue;
        s = jumpToElement(elArrDH, i);
        MCHECK(s, NODENAME + "ERROR setting outputs, could not jump to element: " + std::to_string(i).c_str());
        StrataName elName = opPtr->elementsAdded[i];
        if (elName.empty()) {
            l("element " + str(i) + " has empty name, skipping");
        }
        l("test error name");
        l(elName);
        l(elName.c_str());
        elArrDH.outputValue().child(aStNameOut).setString(MString(elName.c_str()));
        SElement* el = manifold.getEl(elName);
        
        elArrDH.outputValue().child(aStGlobalIndex).setInt(
            el->globalIndex
        );
        elArrDH.outputValue().child(aStTypeOut).setInt(
            el->elType);
        elArrDH.outputValue().child(aStElTypeIndex).setInt(
            el->elIndex);

        /* matrices*/
        elArrDH.outputValue().child(aStPointFinalWorldMatrixOut).setMMatrix(
            toMMatrix(manifold.pDataMap[elName].finalMatrix)
        );

        ///* parent info*/
        //MArrayDataHandle parentDH = elArrDH.outputValue().child(aStSpace)
        //switch (el->elType) {
        //case StrataElType::point :{
        //
        //    SPointData& pData = manifold.pDataMap[el->name];
        //    for (int n = 0; n < static_cast<int>(pData.spaceDatas.size()); n++) {
        //        
        //    }
        //    }
        //}
    }

    /* set everything clean*/
    data.setClean(aStElement);
    data.setClean(aStGlobalIndex);
    data.setClean(aStElTypeIndex);
    data.setClean(aStTypeOut);
    data.setClean(aStOutput);
    data.setClean(aStPointWorldMatrixIn);

    return MS::kSuccess;
}


//MStatus StrataElementOpNode::legalConnection(
//    const MPlug& plug,
//    const MPlug& otherPlug,
//    bool 	asSrc,
//    bool& isLegal
//) {
//    /* check if the incoming plug is the strataGraph connection -
//    * if so, check that the incoming node is a StrataGraphNode
//
//    asSrc	is this plug a source of the connection
//    the docs and argument names around plug connection direction are riddles
//    */
//    MStatus s = StrataOpNodeBase::legalConnection(plug, otherPlug, asSrc, isLegal);
//    if (s == MS::kSuccess) {
//        return s; // already treated 
//    }
//    
//
//    return MS::kUnknownParameter;
//}

