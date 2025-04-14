
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


    aStData = cFn.create("stData", "stData");
    cFn.setArray(true);

    aStExp = tFn.create("stExp", "stExp", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    cFn.addChild(aStExp);

    std::vector<MObject> drivers{
        aStData,
        aStExp
    };
    std::vector<MObject> driven{
    };

    std::vector<MObject> toAdd{
        aStData
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
    //DEBUGS("element postConstructor");
    //superT::postConstructor(thisMObject());
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

    DEBUGS("shape compute")
        MS s(MS::kSuccess);

    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }

    // pass to bases
    //s = superT::compute(thisMObject(), plug, data);
    MCHECK(s, NODENAME + " ERROR in strata bases compute, halting");

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

