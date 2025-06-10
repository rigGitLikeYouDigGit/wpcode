
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
    cFn.setArray(true);
    aStExpIn = tFn.create("stExpIn", "stExpIn", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStExpIn);
    aStSpaceModeIn = eFn.create("stSpaceModeIn", "stSpaceModeIn", 0);
    eFn.addField("world", 0); // snap element to data specified
    eFn.addField("localOnFinal", 1); // apply param in local space on top of final element
    eFn.addField("uvn", 2); // modify UVN in parent space by local vector - how does this work with multiple spaces?
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

    aStDataOut = cFn.create("stDataOut", "stDataOut");
    cFn.setArray(true);
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

MStatus StrataShapeNode::editPrevOpData(
    MObject& nodeObj, MDataBlock& data, MDataHandle& elDH,
    StrataManifold* manifold, SElement* finalEl, StrataOp* targetOp
) {
    /* finalEl is element in latest value of graph
    * 
    * for more complex ops, we might need to have invert() method
    * for each one - here just modify the data for now
    */
    MStatus s;

    int spaceIndex = getSpaceIndex<StrataShapeNode>(nodeObj, data, elDH,
        manifold, finalEl, targetOp);
    if (spaceIndex == -2) {
        DEBUGSL("specified space that doesn't exist for " + finalEl->name)
        return s;
    }

    switch (finalEl->elType) {
        case StrataElType::point: {
            // check data is found
            //Affine3f tf = toAff(MFnMatrixData(elDH.child(aStMatrixIn).data()).matrix());
            Affine3f tf = toAff(elDH.child(aStMatrixIn).asMatrix());
            auto elDataPair = targetOp->opPointDataMap.find(finalEl->name);
            if (elDataPair == targetOp->opPointDataMap.end()) {
                DEBUGS("could not find data for point " + finalEl->name + " in op " + targetOp->name);
                return s;
            }
            
            int spaceMode = elDH.child(aStSpaceModeIn).asInt();
            switch (spaceMode) {
            case 0: { // worldspace snap

                break;
            }
            }

        }
    }
    
}


MStatus StrataShapeNode::syncStrataParams(MObject& nodeObj, MDataBlock& data) {
    /* pull in any attributes to update element information in graph -
    * 
    * look up latest (???) node in element's history and modify its data
    * latest node is this shape/merge op, which hasn't run yet - 
    * - need to check incoming streams in reverse order for element name
    * - look at that el's history
    * - get op in graph
    * - edit 
    * - add this op to history
    * 
    * later modifications will send deltas to this op by default
    * or not? maybe we try to send as far back as possible instead
    */
    MS s;
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
        SElement* el;
        StrataManifold& manifold = StrataManifold();
        thisStrataOpT* opPtr = getStrataOp<StrataShapeNode>(data);
        if (opPtr == nullptr) {
            DEBUGSL("SHAPE NODE OP PTR IS NULL ")
                return MS::kFailure;
        }

        StrataOp* targetNode = nullptr;
        // check through op inputs
        for (auto& n : opPtr->inputNodes()) {
            StrataOp* inOpPtr = static_cast<StrataOp*>(n);
            el = inOpPtr->value().getEl(inExpStr.asChar());
            if (el != nullptr) {
                manifold = inOpPtr->value();
                break;
            }
        }
        // if no element found of this name, just move on
        if (el == nullptr) {
            continue;
        }
        
        StrataOp* opToModify = nullptr;
        // get element history
        std::string opName = el->opHistory[0];
        opToModify = opPtr->getGraphPtr()->getNode<StrataOp>(opName);
        if (opToModify == nullptr) { // this shouldn't be possible
            DEBUGSL("element " + el->name + "modified by node " + opName + " which was not found in graph")
            return MS::kFailure;
        }
        // finally actually change the saved op data, from this 
        editPrevOpData(nodeObj, data, elDH,
            manifold, el, opToModify);

        // flag op as dirty
        opToModify->dirtyMap["main"] = true;
        opPtr->getGraphPtr()->nodePropagateDirty(opToModify->index);
    }

    return s;
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

    ///*/////////// copying eval logic from parent class ////////////*/
    //auto& nodeObj = thisMObject();
    //if (plug.attribute() == aStOpNameOut) {
    //    syncOpNameOut<StrataShapeNode>(nodeObj, data);
    //    data.setClean(plug);
    //    return s;
    //}

    //// copy graph data if it's dirty
    //if (!data.isClean(aStInput)) {
    //    s = syncIncomingGraphData<StrataShapeNode>(nodeObj, data);
    //    DEBUGS("syncIncoming graph complete")
    //        MCHECK(s, "error syncing incoming graph data");
    //}
    //s = syncStrataParams(nodeObj, data);
    //DEBUGSL("base synced strata params")
    //    MCHECK(s, "error syncing strata params");

    //Status graphS;
    //int upToNode = getOpIndexFromNode<StrataShapeNode>(data);
    //DEBUGSL("base got index from node")
    //    opGraphPtr.get()->evalGraph(graphS, upToNode);
    //DEBUGSL("base graph eval'd");

    //data.setClean(aStOutput);
    //data.setClean(aStOpName);
    ////////////////////////


    /* pull in drawing values to cache*/
    pointOpacity = data.inputValue(aStShowPoints).asFloat();



    data.setClean(plug);

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




