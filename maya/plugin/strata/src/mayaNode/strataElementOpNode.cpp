
#pragma once
#include <vector>
//#include <array>



#include "../macro.h"
#include "../api.h"
#include "strataElementOpNode.h"
#include "strataGraphNode.h"
#include "strataPointNode.h"
#include "strataOpNodeBase.h"
#include "../lib.cpp"
#include "../strataop/strataElementOp.h"

using namespace ed;



MTypeId StrataElementOpNode::kNODE_ID(0x00122CA2);
MString StrataElementOpNode::kNODE_NAME("strataElementOp");

MString StrataElementOpNode::drawDbClassification("drawdb/geometry/strataElementOp");
MString StrataElementOpNode::drawRegistrantId("StrataElementOpNodePlugin");


//MObject StrataElementOpNode::aStPoint;
//MObject StrataElementOpNode::aStPointLocalMatrix;
//MObject StrataElementOpNode::aStPointHomeMatrix;
//MObject StrataElementOpNode::aStPointName;
//MObject StrataElementOpNode::aStResult;


DEFINE_STRATA_STATIC_MOBJECTS(StrataElementOpNode)


DEFINE_STATIC_NODE_MEMBERS(
    STRATAADDPOINTSOPNODE_STATIC_MEMBERS, StrataElementOpNode
)

//StrataElementOpNode::addStrataAttrs;

MStatus StrataElementOpNode::initialize() {
    MStatus s = MS::kSuccess;
    MFnNumericAttribute nFn;
    MFnCompoundAttribute cFn;
    MFnEnumAttribute eFn;
    MFnMatrixAttribute mFn;
    MFnGenericAttribute gFn;
    MFnMessageAttribute msgFn;
    MFnTypedAttribute tFn;
     

    cFn.setObject(aStElement);

    aStExp = tFn.create("stExp", "stExp", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    aStType = eFn.create("stType", "stType", 0);
    eFn.addField("point", 0);
    eFn.addField("edge", 1);
    eFn.addField("face", 2);
    aStName = tFn.create("stName", "stName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    aStGlobalIndex = nFn.create("stGlobalIndex", "stGlobalIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);
    aStElTypeIndex = nFn.create("stElTypeIndex", "stElTypeIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);

    // point attributes
    aStPointInWorldMatrix = mFn.create("stPointInWorldMatrix", "stPointInWorldMatrix");
    mFn.setDefault(MMatrix());
    aStPointOutFinalDriverMatrix = mFn.create("stPointOutFinalDriverMatrix", "stPointOutFinalDriverMatrix");
    mFn.setDefault(MMatrix());
    aStPointOutFinalLocalOffsetMatrix = mFn.create("stPointOutFinalLocalOffsetMatrix", "stPointOutFinalLocalOffsetMatrix");
    mFn.setDefault(MMatrix());
    aStPointOutFinalWorldMatrix = mFn.create("stPointOutFinalWorldMatrix", "stPointOutFinalWorldMatrix");
    mFn.setDefault(MMatrix());

    // edge attributes
    aStEdgeResolution = nFn.create("stEdgeResolution", "stEdgeResolution", MFnNumericData::kInt, 1);
    nFn.setMin(1);
    aStEdgeNormaliseParam = nFn.create("stEdgeNormaliseParam", "stEdgeNormaliseParam", MFnNumericData::kFloat, 1.0);
    nFn.setMin(0.0);
    nFn.setMin(1.0);
    aStEdgeUseLength = nFn.create("stEdgeUseLength", "stEdgeUseLength", MFnNumericData::kFloat, 1.0);
    nFn.setMin(0.0);
    nFn.setMin(1.0);
    aStEdgeReverse = nFn.create("stEdgeReverse", "stEdgeReverse", MFnNumericData::kFloat, 0.0);
    nFn.setMin(0.0);
    nFn.setMin(1.0);
    // edge start/end entries - unsure if these should be all one array instead
    aStEdgeStartIndex = nFn.create("stEdgeStartIndex", "stEdgeStartIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);
    aStEdgeStartName = tFn.create("stEdgeStartName", "stEdgeStartName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    
    aStEdgeEndIndex = nFn.create("stEdgeEndIndex", "stEdgeEndIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);
    aStEdgeEndName = tFn.create("stEdgeEndName", "stEdgeEndName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));

    MFnCompoundAttribute edgeMidFn;
    aStEdgeMid = edgeMidFn.create("stEdgeMid", "stEdgeMid");
    edgeMidFn.setArray(true);
    edgeMidFn.setUsesArrayDataBuilder(true);
    aStEdgeMidIndex = nFn.create("stEdgeMidIndex", "stEdgeMidIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);
    aStEdgeMidName = tFn.create("stEdgeMidName", "stEdgeMidName", MFnData::kString);
    tFn.setDefault(MFnStringData().create(""));
    edgeMidFn.addChild(aStEdgeMidIndex);
    edgeMidFn.addChild(aStEdgeMidName);

    // face attributes
    MFnCompoundAttribute faceDriverFn;
    aStFaceDriver = faceDriverFn.create("stFaceDriver", "stFaceDriver");
    faceDriverFn.setArray(true);
    faceDriverFn.setUsesArrayDataBuilder(true);
    aStFaceDriverIndex = nFn.create("stFaceDriverIndex", "stFaceDriverIndex", MFnNumericData::kInt, -1);
    nFn.setMin(-1);
    aStFaceDriverName = tFn.create("stFaceDriverName", "stFaceDriverName", MFnData::kString);
    faceDriverFn.addChild(aStFaceDriverIndex);
    faceDriverFn.addChild(aStFaceDriverName);
    
    addChildAttributes(cFn, 
        { &aStExp, &aStType, &aStName, &aStGlobalIndex, &aStElTypeIndex,
        &aStPointInWorldMatrix, &aStPointOutFinalDriverMatrix, &aStPointOutFinalLocalOffsetMatrix, &aStPointOutFinalWorldMatrix,
        &aStEdgeResolution, &aStEdgeNormaliseParam, &aStEdgeUseLength}
    );

    std::vector<MObject> drivers;
    std::vector<MObject> driven;

    s = addStrataAttrs(drivers, driven);
    MCHECK(s, "could not add Strata attrs");

    addAttributes<StrataElementOpNode>(drivers);
    addAttributes<StrataElementOpNode>(driven);
    setAttributesAffect<StrataElementOpNode>(drivers, driven);

    CHECK_MSTATUS_AND_RETURN_IT(s);
    return s;
}


MStatus StrataElementOpNode::compute(const MPlug& plug, MDataBlock& data) {
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


MStatus StrataElementOpNode::legalConnection(
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

