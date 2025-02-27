
#pragma once
#include <vector>
//#include <array>



#include "../macro.h"
#include "../api.h"
#include "strataAddPointsOpNode.h"
#include "strataGraphNode.h"
#include "../lib.cpp"


using namespace ed;



MTypeId StrataAddPointsOpNode::kNODE_ID(0x00122CA2);
MString StrataAddPointsOpNode::kNODE_NAME("strataAddPointsOp");

MString StrataAddPointsOpNode::drawDbClassification("drawdb/geometry/strataAddPointsOp");
MString StrataAddPointsOpNode::drawRegistrantId("StrataAddPointsOpNodePlugin");

DEFINE_STRATA_STATIC_MOBJECTS(StrataAddPointsOpNode)


MStatus StrataAddPointsOpNode::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;

    // driver array
    std::vector<MObject> drivers;
    std::vector<MObject> driven;
    s = addStrataAttrs<StrataAddPointsOpNode>(drivers, driven);
    MCHECK(s, "could not add Strata attrs");
    addAttributes<StrataAddPointsOpNode>(drivers);
    addAttributes<StrataAddPointsOpNode>(driven);
    setAttributesAffect<StrataAddPointsOpNode>(drivers, driven);
    

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataAddPointsOpNode::compute(const MPlug& plug, MDataBlock& data) {
    MS s(MS::kSuccess);

    if (data.isClean(plug)) {
        return s;
    }

    data.setClean(plug);

    return s;
}

void StrataAddPointsOpNode::postConstructor() {

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

}

