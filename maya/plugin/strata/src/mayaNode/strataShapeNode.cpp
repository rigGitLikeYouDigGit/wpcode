
#pragma once
#include <vector>
//#include <array>

#include "../macro.h"
#include "../api.h"
#include "../MInclude.h"

#include "strataShapeNode.h"
#include "strataGraphNode.h"
#include "strataPointNode.h"
#include "strataOpNodeBase.h"
#include "../lib.h"
#include "../stringLib.h"
#include "../strataop/elementOp.h"

using namespace ed;

MTypeId StrataShapeNode::kNODE_ID(0x00122CA3);
MString StrataShapeNode::kNODE_NAME("strataShape");

MString StrataShapeNode::drawDbClassification("drawdb/geometry/strataShape");
MString StrataShapeNode::drawRegistrantId("StrataShape");

//DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataShapeNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataShapeNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(NODE_STATIC_MEMBERS, StrataShapeNode)


MStatus StrataShapeNode::initialize() {
    DEBUGSL("shape initialize")
        MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;


    aStDataIn = cFn.create("stDataIn", "stDataIn");
    cFn.setUsesArrayDataBuilder(true);
    aStExpIn = tFn.create("stExpIn", "stExpIn", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStExpIn);
    aStSpaceModeIn = eFn.create("stSpaceModeIn", "stSpaceModeIn", 0);
    eFn.addField("local", 0); // snap element to data specified
    eFn.addField("world", 1); // apply param in local space on top of final element
    cFn.addChild(aStSpaceModeIn);
    aStSpaceNameIn = tFn.create("stSpaceNameIn", "stSpaceNameIn", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStSpaceNameIn);
    aStSpaceIndexIn = nFn.create("stSpaceIndexIn", "stSpaceIndexIn", MFnNumericData::kInt, 0);
    nFn.setMin(0);
    cFn.addChild(aStSpaceIndexIn);
    aStMatrixIn = mFn.create("stMatrixIn", "stMatrixIn");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aStMatrixIn);
    // you wouldn't specify a local UVN and a world matrix within the same entry, second mode attr not needed
    //aStUVNModeIn = eFn.create("stUVNModeIn", "stUVNModeIn", 0); 
    //eFn.addField("local", 0); // snap element to data specified
    //eFn.addField("world", 1); // apply param in local space on top of final element
    //cFn.addChild(aStUVNModeIn);
    aStUVNIn = nFn.createPoint("stUVNIn", "stUVNIn");
    cFn.addChild(aStUVNIn);

    aStDataOut = cFn.create("stDataOut", "stDataOut");
    cFn.setArray(true);
    cFn.setUsesArrayDataBuilder(true);
    aStExpOut = tFn.create("stExpOut", "stExpOut", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStExpOut);
    aStMatrixOut = mFn.create("stMatrixOut", "stMatrixOut");
    mFn.setDefault(MMatrix::identity);
    cFn.addChild(aStMatrixOut);
    aStCurveOut = tFn.create("stCurveOut", "stCurveOut", MFnData::kNurbsCurve);
    tFn.setDefault(MFnNurbsCurveData().create());
    cFn.addChild(aStMatrixOut);

    aStShowPoints = nFn.create("stShowPoints", "stShowPoints", MFnNumericData::kFloat, 1.0);
    nFn.setChannelBox(true);
    nFn.setMin(0.0);
    nFn.setMax(1.0);

    std::vector<MObject> drivers{
        aStExpIn,
        aStSpaceModeIn,
        aStMatrixIn,
        aStUVNIn,
        aStSpaceNameIn,
        aStSpaceIndexIn,

                aStExpOut,

        aStShowPoints

    };
    std::vector<MObject> driven{
        aStMatrixOut,
        aStCurveOut,
        
    };

    std::vector<MObject> toAdd{
        aStDataIn,
        aStDataOut,
        aStShowPoints
    };

    s = addStrataAttrs<thisT>(drivers, driven, toAdd);
    MCHECK(s, "could not add Strata attrs to StrataShape");

    addAttributes<thisT>(toAdd);
    setAttributesAffect<thisT>(drivers, driven);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    DEBUGS("end shape initialize");
        return s;
}



void StrataShapeNode::postConstructor() {
    /* no post needed*/
}

MStatus StrataShapeNode::legalConnection(
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

MStatus StrataShapeNode::connectionMade(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc
) {
    return StrataOpNodeBase::connectionMade<StrataShapeNode>(
        thisMObject(),
        plug,
        otherPlug,
        asSrc
    );
    //return MPxNode::connectionMade(
    //    plug,
    //    otherPlug,
    //    asSrc
    //);
}

MStatus StrataShapeNode::connectionBroken(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc
) {
    //DEBUGSL("el connection broken")
        return StrataOpNodeBase::connectionBroken<StrataShapeNode>(
            thisMObject(),
            plug,
            otherPlug,
            asSrc
        );
    /*return MPxNode::connectionBroken(
        plug,
        otherPlug,
        asSrc
    );*/
}

template <typename NodeT>
int getSpaceIndex(
    MObject& nodeObj, MDataBlock& data, MDataHandle& elDH,
    StrataManifold* manifold, SElement* finalEl, StrataOp* targetOp
) { /* return -2 to say space is missing
    -1 to say use first or no space
    
    TODO: move to a manifold function
    */
    int spaceIndex = -1;
    bool foundSpace = false;
    // get which space to affect
    MString spaceName = elDH.child(NodeT::aStSpaceNameIn).asString();
    if (!spaceName.isEmpty()) {
        for (auto i : finalEl->spaces) {
            spaceIndex += 1; // will start at 0
            SElement* spaceEl = manifold->getEl(i);
            if (spaceEl == nullptr) {
                /* gave name but not found in driving spaces*/
                return -2;
            }
            
            if (std::strcmp(spaceEl->name.c_str(), spaceName.asChar())) { // found matching name
                return spaceIndex;
            }
        }
    }
    spaceIndex = elDH.child(NodeT::aStSpaceIndexIn).asInt();
    if (spaceIndex > (finalEl->spaces.size() - 1)) {
        return -2; // index greater than number of spaces
    }
    return spaceIndex;
}

MStatus StrataShapeNode::addDeltaTarget(
    MObject& nodeObj, MDataBlock& data, MDataHandle& elDH,
    ed::StrataManifold& manifold, ed::SElement* finalEl, ed::SAtomBackDeltaGroup& deltaGrp
) {
    /* 
    * add a delta to be matched - 
    * 
    * for merge op, only add new elements from copied graphs.
    * then for backpropagation, any deltas required of those elements
    * are transferred directly into the incoming aux streams
    */
    MStatus s;

    //int spaceIndex = getSpaceIndex<StrataShapeNode>(nodeObj, data, elDH,
    //    manifold, finalEl, targetOp);
    //if (spaceIndex == -2) {
    //    DEBUGSL("specified space that doesn't exist for " + finalEl->name)
    //    return s;
    //}
    SAtomMatchTarget matchTarget;
    switch (finalEl->elType) {
        case StrataElType::point: {
            // check data is found
            //Affine3f tf = toAff(MFnMatrixData(elDH.child(aStMatrixIn).data()).matrix());
            Affine3f tf = toAff(elDH.child(aStMatrixIn).asMatrix());
            //matchTarget. = finalEl->globalIndex;
            matchTarget.matrix = tf;
            
            int spaceMode = elDH.child(aStSpaceModeIn).asInt();
            matchTarget.matrixMode = spaceMode;

        }
    }
    deltaGrp.targetMap[finalEl->name].push_back(matchTarget);
    return s;
}


MStatus StrataShapeNode::syncStrataParams(MObject& nodeObj, MDataBlock& data) {
    /* gather any parametres needed to do merge of incoming streams
    */
    MS s;


    return s;
}

MStatus StrataShapeNode::runShapeBackPropagation(MObject& nodeObj, MDataBlock& data) {

    MS mStat;
    // check for DELTAS, or directed changes to graph data
    MArrayDataHandle inArrDH = data.inputArrayValue(aStDataIn);

    // build up a group of deltas to match
    SAtomBackDeltaGroup deltaGrp;

    for (unsigned int i = 0; i < inArrDH.elementCount(); i++) {
        inArrDH.jumpToArrayElement(i);
        MDataHandle elDH = inArrDH.inputValue();
        MString inExpStr = elDH.child(aStExpIn).asString();
        if (inExpStr.isEmpty()) {
            continue;
        }
        StrataManifold& manifold = StrataManifold();
        thisStrataOpT* opPtr = getStrataOp<StrataShapeNode>(data);
        if (opPtr == nullptr) {
            DEBUGSL("SHAPE NODE OP PTR IS NULL ")
                return MS::kFailure;
        }

        manifold = opPtr->value();
        auto el = manifold.getEl(inExpStr.asChar());
        // if no element found of this name, just move on
        if (el == nullptr) {
            continue;
        }

        // add delta from data handle
        addDeltaTarget(thisMObject(), data, elDH,
            manifold, el, deltaGrp);
    }

    if (!deltaGrp.targetMap.size()) {
        DEBUGSL("no targets gathered in shape node, skipping");
        return mStat;
    }

    // back-propagate errors to re-eval the graph with any dirty nodes re-eval'd
    StrataMergeOp* opPtr = getStrataOp<StrataShapeNode>(data);
    Status st;
    StrataAuxData auxData;
    st = opPtr->runBackPropagation(
        st,
        opPtr,
        opPtr->value(),
        deltaGrp,
        auxData
    );

    if (st.val) {
        DEBUGSL("ERROR IN BACKPROPAGATION") ;
        DEBUGS(st.msg);
        return MStatus::kFailure;
    }

    return mStat;
}

MStatus StrataShapeNode::compute(const MPlug& plug, MDataBlock& data) {
    /* pull in any data overrides set as this node's params,
    then populate out attributes with data from op graph

    If shape node is hidden, only evaluate ops in history of expOut attributes.
    If visible, need to eval the whole graph
    */
    MS s(MS::kSuccess);
    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }
    DEBUGS("shape compute");
    // run strata op merge
    superT::compute<StrataShapeNode>(thisMObject(), plug, data);

    // back propagate if necessary
    runShapeBackPropagation(thisMObject(), data);

    /* pull in drawing values to cache*/
    pointOpacity = data.inputValue(aStShowPoints).asFloat();

    s = populateOutputs(data);

    data.setClean(plug);

    return s;
}


MStatus StrataShapeNode::populateOutputs(MDataBlock& data) {
    /* for each output compound entry, look at which element it's
    requesting, and populate it
    consider also adding space data?
    */
    MStatus s = MS::kSuccess;

    thisStrataOpT* opPtr = getStrataOp<StrataShapeNode>(data);

    MArrayDataHandle outArrDH = data.outputArrayValue(aStDataOut);
    for (unsigned int i = 0; i < outArrDH.elementCount(); i++) {
        s = jumpToPhysicalIndex(outArrDH, i);
        MCHECK(s, "error jumping to out arr element " + std::to_string(i));

        MString expOut = outArrDH.outputValue(&s).child(aStExpOut).asString();
        MCHECK(s, "error retrieving expOut from arr element " + std::to_string(i));

        if (expOut.isEmpty()) { continue; } // skip if empty

        // find matching element in this node's value manifold
        SElement* el = opPtr->value().getEl(expOut.asChar());
        if (el == nullptr) { // skip if not found
            continue;
        }

        switch (el->elType) {
        case StrataElType::point :{ // set matrices
            SPointData& d = opPtr->value().pDataMap[el->name];
            outArrDH.outputValue().child(aStMatrixOut).setMMatrix(
                toMMatrix(d.finalMatrix)
            );
            break;
            }
        case StrataElType::edge: {
            break;
        }
        case StrataElType::face: {
            break;
        }
        }

    }

    return s;
}

//MStatus StrataShapeNode::legalConnection(
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

//MTypeId StrataShapeGeometryOverride::id = MTypeId(0x8003D);




