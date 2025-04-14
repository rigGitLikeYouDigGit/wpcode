
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
//#include "strataOpNodeBase.cpp"
#include "../lib.cpp"
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

    /*expression to generate given elements - leave blank for raw points
    */
    aStDriverExp = tFn.create("stDriverExp", "stDriverExp", MFnData::kString); 
    tFn.setDefault(MFnStringData().create(""));

    /* expression for parents of given elements*/
    aStParentExp = tFn.create("stParentExp", "stParentExp", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));

    /* name of new element to create*/
    aStName = tFn.create("stName", "stName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    /* global index of new element created */ 
    aStGlobalIndex = nFn.create("stGlobalIndex", "stGlobalIndex", MFnNumericData::kInt, -1);
    nFn.setKeyable(false);
    nFn.setMin(-1);
    nFn.setWritable(false);
    /* component-type-specific index of element created */
    aStElTypeIndex = nFn.create("stElTypeIndex", "stElTypeIndex", MFnNumericData::kInt, -1);
    nFn.setKeyable(false);
    nFn.setMin(-1);
    nFn.setWritable(false);

    cFn.addChild(aStDriverExp);
    cFn.addChild(aStParentExp);
    cFn.addChild(aStName);
    cFn.addChild(aStGlobalIndex);
    cFn.addChild(aStElTypeIndex);

    aStTypeOut = eFn.create("stTypeOut", "stTypeOut", 0);
    eFn.addField("point", 0);
    eFn.addField("edge", 1);
    eFn.addField("face", 2);
    cFn.addChild(aStTypeOut);

    //// element attributes
    aStMatchWorldSpaceIn = nFn.create("stMatchWorldSpaceIn", "stMatchWorldSpaceIn", MFnNumericData::kFloat, 1.0);
    cFn.addChild(aStMatchWorldSpaceIn);
    aStDriverWeightIn = nFn.create("stDriverWeightIn", "stInDriverWeightIn", MFnNumericData::kFloat, 1.0);
    nFn.setArray(true);
    cFn.addChild(aStDriverWeightIn);

    // point attributes
    aStPointWorldMatrixIn = mFn.create("stPointWorldMatrixIn", "stPointWorldMatrixIn");
    mFn.setDefault(MMatrix());
    cFn.addChild(aStPointWorldMatrixIn);

    aStPointDriverLocalMatrixIn = mFn.create("stPointDriverLocalMatrixIn", "stPointDriverLocalMatrixIn");
    mFn.setDefault(MMatrix()); /* separate local matrix per driver*/
    mFn.setArray(true);
    cFn.addChild(aStPointDriverLocalMatrixIn);

    aStPointFinalWorldMatrixOut = mFn.create("stPointFinalWorldMatrixOut", "stPointFinalWorldMatrixOut");
    mFn.setDefault(MMatrix());
    mFn.setWritable(false);
    cFn.addChild(aStPointFinalWorldMatrixOut);

    aStPointWeightedDriverMatrixOut = mFn.create("stPointWeightedDriverMatrixOut", "stPointWeightedDriverMatrixOut");
    mFn.setDefault(MMatrix());
    cFn.addChild(aStPointWeightedDriverMatrixOut);
    aStPointWeightedLocalOffsetMatrixOut = mFn.create("stPointWeightedLocalOffsetMatrixOut", "stPointWeightedLocalOffsetMatrixOut");
    mFn.setDefault(MMatrix());
    mFn.setArray(true);
    cFn.addChild(aStPointWeightedLocalOffsetMatrixOut);
    aStPointDriverMatrixOut = mFn.create("stPointDriverMatrixOut", "stPointDriverMatrixOut");
    mFn.setDefault(MMatrix());
    mFn.setArray(true);
    cFn.addChild(aStPointDriverMatrixOut);


    //// edge attributes
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
        aStParentExp,

        aStMatchWorldSpaceIn,
        aStDriverWeightIn,

        aStPointWorldMatrixIn,
        aStPointDriverLocalMatrixIn,
        aStPointWeightedDriverMatrixOut,
        aStPointWeightedLocalOffsetMatrixOut,
        aStPointFinalWorldMatrixOut,
    };
    std::vector<MObject> driven{
        aStGlobalIndex,
        aStElTypeIndex,
        aStTypeOut,

        aStPointFinalWorldMatrixOut,

        aStEdgeCurveOut
    };

    std::vector<MObject> toAdd{
        aStElement
    };

    s = addStrataAttrs<thisT>(drivers, driven, toAdd);
    MCHECK(s, "could not add Strata attrs");

    addAttributes<thisT>(toAdd);
    setAttributesAffect<thisT>(drivers, driven);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    DEBUGS("end element initialize")
    return s;
}



void StrataElementOpNode::postConstructor() {
    DEBUGS("element postConstructor");
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
    DEBUGSL("el connection made")
        MStatus s = superT::connectionMade<thisT>(
            thisMObject(),
            plug,
            otherPlug,
            asSrc
        );
    return MPxNode::connectionMade(
        plug,
        otherPlug,
        asSrc
    );
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

MStatus StrataElementOpNode::syncStrataParams(MObject& nodeObj, MDataBlock& data) {
    /* build map of element names to expressions, from maya node
    */
    MS s;
    thisStrataOpT* opPtr = getStrataOp<thisT>(data);
    if (opPtr == nullptr) {
        MGlobal::displayError(NODENAME + "COULD NOT RETRIEVE OP to sync");
        //NODELOG("COULD NOT RETRIEVE OP to sync");
        return MS::kFailure;
    }

    opPtr->namePointDataMap.clear();

    // check names currently found, remove any missing
    std::unordered_set<std::string> foundNames;

    MArrayDataHandle elDH = data.inputArrayValue(aStElement, &s);
    MCHECK(s, NODENAME + "error getting input array value");
    for (unsigned int i = 0; i < elDH.elementCount(); i++) {
        s = jumpToElement(elDH, i);
        MCHECK(s, "could not jump to element" + std::to_string(i));
        //MDataHandle iDH = 
        MDataHandle nameDH = elDH.inputValue().child(aStName);
        MDataHandle driverExpDH = elDH.inputValue().child(aStDriverExp);

        // check if name is empty
        std::string elName(nameDH.asString().asChar());
        trimEnds(elName);
        if (elName.empty()) { // empty name
            continue;
        }

        // driver exp
        std::string driverExpStr(driverExpDH.asString().asChar());
        foundNames.insert(elName);
        if (opPtr->paramNameExpMap.count(elName)) { // name found, check if exp text is different
            if (opPtr->paramNameExpMap[elName].srcStr == driverExpStr) { // matches, all fine
            }
            else { // text has changed, recompile
                opPtr->paramNameExpMap[elName].setSource(driverExpStr.c_str());
            }
        }
        else { // not found, make new expression
            opPtr->paramNameExpMap[elName] = expns::Expression(driverExpStr.c_str());
        }

        // parent exp
        std::string parentExpStr = elDH.inputValue().child(aStParentExp).asString().asChar();
        std::string elParentName = elName + "!";
        foundNames.insert(elParentName);
        if (opPtr->paramNameExpMap.count(elParentName)) { // name found, check if exp text is different
            if (opPtr->paramNameExpMap[elParentName].srcStr == driverExpStr) { // matches, all fine
            }
            else { // text has changed, recompile
                opPtr->paramNameExpMap[elParentName].setSource(driverExpStr.c_str());
            }
        }
        else { // not found, make new expression
            opPtr->paramNameExpMap[elParentName] = expns::Expression(driverExpStr.c_str());
        }

        // remove any params from op not found in names
        for (auto& i : opPtr->paramNameExpMap) {
            if (foundNames.find(i.first) == foundNames.end()) {
                opPtr->paramNameExpMap.erase(i.first);
            }
        }
        // pull in element data
        float matchWorldSpace = elDH.inputValue().child(aStMatchWorldSpaceIn).asFloat();
        // always match in worldspace for now
        switch (elName[0]) {
            case 'p': { // points have no drivers
                opPtr->namePointDataMap[elName].finalMatrix = elDH.inputValue().child(aStPointWorldMatrixIn).asMatrix();
                break;
            }
            case 'e': {
                
            }
        }
        
    }
    return s;
}

MStatus StrataElementOpNode::compute(const MPlug& plug, MDataBlock& data) {

    DEBUGS("element compute")
        MS s(MS::kSuccess);

    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }

    // pass to bases, compute strata op
    s = superT::compute<thisT>(thisMObject(), plug, data);
    MCHECK(s, NODENAME + " ERROR in strata bases compute, halting");

    // update index attrs from op elements
    thisStrataOpT* opPtr = getStrataOp<thisT>(data);
    MArrayDataHandle elArrDH = data.outputArrayValue(aStElement);
    for (unsigned int i = 0; i < elArrDH.elementCount(); i++){
        jumpToElement(elArrDH, i);
        int elGlobalIndex = opPtr->elementsAdded[i];
        elArrDH.outputValue().child(aStGlobalIndex).setInt(
            elGlobalIndex
        );
        elArrDH.outputValue().child(aStTypeOut).setInt(
            opPtr->value()->getEl(elGlobalIndex)->elType);
        elArrDH.outputValue().child(aStElTypeIndex).setInt(
            opPtr->value()->getEl(elGlobalIndex)->elIndex);

    }

    /* set everything clean*/
    data.setClean(aStElement);
    data.setClean(aStGlobalIndex);
    data.setClean(aStElTypeIndex);
    data.setClean(aStTypeOut);
    return s;
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

