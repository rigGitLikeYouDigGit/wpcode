
#pragma once
#include <vector>
//#include <array>



#include "../macro.h"
#include "../api.h"
#include "strataAddPointsOpNode.h"
#include "strataGraphNode.h"
#include "strataPointNode.h"
#include "strataOpNodeBase.h"
#include "../lib.cpp"
#include "../strataop/strataAddPointsOp.h"

using namespace ed;



MTypeId StrataAddPointsOpNode::kNODE_ID(0x00122CA2);
MString StrataAddPointsOpNode::kNODE_NAME("strataAddPointsOp");

MString StrataAddPointsOpNode::drawDbClassification("drawdb/geometry/strataAddPointsOp");
MString StrataAddPointsOpNode::drawRegistrantId("StrataAddPointsOpNodePlugin");


//MObject StrataAddPointsOpNode::aStPoint;
//MObject StrataAddPointsOpNode::aStPointLocalMatrix;
//MObject StrataAddPointsOpNode::aStPointHomeMatrix;
//MObject StrataAddPointsOpNode::aStPointName;
//MObject StrataAddPointsOpNode::aStResult;


DEFINE_STRATA_STATIC_MOBJECTS(StrataAddPointsOpNode)


DEFINE_STATIC_NODE_MEMBERS(
    STRATAADDPOINTSOPNODE_STATIC_MEMBERS, StrataAddPointsOpNode
)

//StrataAddPointsOpNode::addStrataAttrs;

MStatus StrataAddPointsOpNode::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;
     
    // add base strataAttrs first
    //std::vector<MObject> drivers{
    //    aStPoint//, // unsure if setting compound parent in attributeAffects is enough
    //};
    /*std::vector<MObject> driven{
        aStResult
    };*/
    std::vector<MObject> drivers;
    std::vector<MObject> driven;

    s = addStrataAttrs<StrataAddPointsOpNode>(drivers, driven);
    // error missing symbol here
    // why 
    // why does intellisense not flag it

    MCHECK(s, "could not add Strata attrs");

    // add point attributes to data
    cFn.setObject(aStElData);

    // add matrix attributes
    aStPointWorldMatrix = mFn.create("stPointWorldMatrix", "stPointWorldMatrix");
    cFn.addChild(aStPointWorldMatrix);

    aStPointFinalDriverOutMatrix = mFn.create("stPointFinalDriverOutMatrix", "stPointFinalDriverOutMatrix");
    cFn.addChild(aStPointFinalDriverOutMatrix);

    aStPointFinalLocalOffsetMatrix = mFn.create("stPointFinalLocalOffsetMatrix", "stPointFinalLocalOffsetMatrix");
    cFn.addChild(aStPointFinalLocalOffsetMatrix);
       
    
    addAttributes<StrataAddPointsOpNode>(drivers);
    addAttributes<StrataAddPointsOpNode>(driven);
    setAttributesAffect<StrataAddPointsOpNode>(drivers, driven);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataAddPointsOpNode::compute(const MPlug& plug, MDataBlock& data) {
    MS s(MS::kSuccess);

    // check if plug is already computed
    if (data.isClean(plug)) {
        return s;
    }
    // check if graph connection has been lost
    if (opGraphPtr.expired()) {
        data.setClean(plug);
        return s;
    }




    return s;
}


MStatus StrataAddPointsOpNode::legalConnection(
    const MPlug& plug,
    const MPlug& otherPlug,
    bool 	asSrc,
    bool& isLegal
) {
    /* check if the incoming plug is the strataGraph connection -
    * if so, check that the incoming node is a StrataGraphNode

    asSrc	is this plug a source of the connection
    the docs and argument names around plug connection direction are riddles
    */
    MStatus s = StrataOpNodeBase::legalConnection(plug, otherPlug, asSrc, isLegal);
    if (s == MS::kSuccess) {
        return s; // already treated 
    }
    

    return MS::kUnknownParameter;
}

