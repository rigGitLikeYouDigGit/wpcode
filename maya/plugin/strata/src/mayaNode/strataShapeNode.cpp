
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
//#include "strataOpNodeBase.cpp"
#include "../lib.cpp"
#include "../stringLib.h"
#include "../strataop/elementOp.h"

using namespace ed;

MTypeId StrataShapeNode::kNODE_ID(0x00122CA3);
MString StrataShapeNode::kNODE_NAME("strataShape");

MString StrataShapeNode::drawDbClassification("drawdb/geometry/strataShape");
MString StrataShapeNode::drawRegistrantId("StrataShape");

//DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataShapeNode)
DEFINE_STATIC_NODE_CPP_MEMBERS(STRATABASE_STATIC_MEMBERS, StrataShapeNode);
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
        aStMatrixIn,

                aStExpOut,

        aStShowPoints

    };
    std::vector<MObject> driven{
        aStMatrixOut,
        
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
    //DEBUGSL("el connection made")
    //    MStatus s = superT::connectionMade(
    //        thisMObject(),
    //        plug,
    //        otherPlug,
    //        asSrc
    //    );
    return MPxNode::connectionMade(
        plug,
        otherPlug,
        asSrc
    );
}

MStatus StrataShapeNode::connectionBroken(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc
) {
   /* DEBUGSL("el connection broken")
        MStatus s = superT::connectionBroken(
            thisMObject(),
            plug,
            otherPlug,
            asSrc
        );*/
    return MPxNode::connectionBroken(
        plug,
        otherPlug,
        asSrc
    );
}

MStatus StrataShapeNode::syncStrataParams(MObject& nodeObj, MDataBlock& data) {
    /* no explicit params or op for this, manipulate graph directly
    */
    MS s;
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

    DEBUGS("shape compute")
        

    /* pull in drawing values to cache*/
    pointOpacity = data.inputValue(aStShowPoints).asFloat();

    // pass to bases
    /*s = superT::compute(thisMObject(), plug, data);
    MCHECK(s, NODENAME + " ERROR in strata bases compute, halting");*/



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




