#include "expSet.h"

#include <maya/MFnDependencyNode.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnStringData.h>
#include <maya/MDGMessage.h>
#include <maya/MSelectionList.h>
#include <maya/MGlobal.h>
#include <maya/MFnSet.h>
#include <maya/MPlug.h>
#include <maya/MItDependencyNodes.h>

using namespace wp;

const MTypeId ExpSet::kNODE_ID(0x00123456); // Replace with your studio ID
const MString ExpSet::kNODE_NAME("expSet");
const MString ExpSet::typeName("expSet");

MObject ExpSet::aExpression;
MObject ExpSet::aAutoUpdate;
MObject ExpSet::aBalanceWheel;

ExpSet::ExpSet() 
    : MPxObjectSet()
    , m_isUpdating(false)
{
}

ExpSet::~ExpSet() {
    removeCallbacks();
}

void* ExpSet::creator() {
    return new ExpSet();
}

void ExpSet::postConstructor() {
    MPxObjectSet::postConstructor();
    
    // Set up callback to monitor when nodes are added to the scene
    MStatus status;
    MCallbackId callbackId = MDGMessage::addNodeAddedCallback(
        onNodeAdded,
        kNODE_NAME.asChar(),
        this,
        &status
    );
    
    if (status) {
        m_callbackIds.append(callbackId);
    }
    
    // Perform initial membership update
    updateMembership();
}

MStatus ExpSet::initialize() {
    MStatus status;
    MFnTypedAttribute tAttr;
    MFnNumericAttribute nAttr;

    // Expression attribute - string to match nodes against
    aExpression = tAttr.create("expression", "expr", MFnData::kString, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tAttr.setStorable(true);
    tAttr.setKeyable(false);
    tAttr.setReadable(true);
    tAttr.setWritable(true);
    status = addAttribute(aExpression);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Auto-update attribute - enable/disable automatic membership updates
    aAutoUpdate = nAttr.create("autoUpdate", "auto", MFnNumericData::kBoolean, true, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setStorable(true);
    nAttr.setKeyable(true);
    nAttr.setReadable(true);
    nAttr.setWritable(true);
    status = addAttribute(aAutoUpdate);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Balance wheel - utility attribute to trigger updates
    aBalanceWheel = nAttr.create("balanceWheel", "bw", MFnNumericData::kInt, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setStorable(false);
    nAttr.setKeyable(false);
    nAttr.setReadable(true);
    nAttr.setWritable(false);
    status = addAttribute(aBalanceWheel);
    CHECK_MSTATUS_AND_RETURN_IT(status);

    // Set up attribute relationships
    attributeAffects(aExpression, aBalanceWheel);
    attributeAffects(aAutoUpdate, aBalanceWheel);

    return MS::kSuccess;
}

MStatus ExpSet::updateMembership() {
    if (m_isUpdating) {
        return MS::kSuccess; // Prevent recursive updates
    }
    
    m_isUpdating = true;
    MStatus status;
    
    // Get the expression string
    MPlug exprPlug(thisMObject(), aExpression);
    MString expression;
    exprPlug.getValue(expression);
    
    if (expression.length() == 0) {
        m_isUpdating = false;
        return MS::kSuccess;
    }
    
    // Clear current set membership
    MFnSet setFn(thisMObject(), &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    
    setFn.clear();
    
    // Build selection list from expression using Maya's ls command
    MString command = "ls -long ";
    command += expression;
    
    MStringArray results;
    status = MGlobal::executeCommand(command, results);
    
    if (status) {
        // Add matching nodes to the set
        for (unsigned int i = 0; i < results.length(); i++) {
            MSelectionList selList;
            status = selList.add(results[i]);
            
            if (status) {
                MObject obj;
                status = selList.getDependNode(0, obj);
                
                if (status && !obj.isNull()) {
                    // Add to set
                    setFn.addMember(obj);
                }
            }
        }
    }
    
    m_isUpdating = false;
    return MS::kSuccess;
}

void ExpSet::onNodeAdded(MObject& node, void* clientData) {
    ExpSet* setNode = static_cast<ExpSet*>(clientData);
    if (!setNode) {
        return;
    }
    
    // Check if auto-update is enabled
    MPlug autoUpdatePlug(setNode->thisMObject(), aAutoUpdate);
    bool autoUpdate = false;
    autoUpdatePlug.getValue(autoUpdate);
    
    if (!autoUpdate) {
        return;
    }
    
    // Get the expression
    MPlug exprPlug(setNode->thisMObject(), aExpression);
    MString expression;
    exprPlug.getValue(expression);
    
    if (expression.length() == 0) {
        return;
    }
    
    // Get the name of the newly added node
    MFnDependencyNode nodeFn(node);
    MString nodeName = nodeFn.name();
    
    // Check if the new node matches the expression
    MString command = "ls -long \"";
    command += expression;
    command += "\" \"";
    command += nodeName;
    command += "\"";
    
    MStringArray results;
    MStatus status = MGlobal::executeCommand(command, results);
    
    // If the node is in the results, add it to the set
    if (status && results.length() > 0) {
        for (unsigned int i = 0; i < results.length(); i++) {
            if (results[i] == nodeName || results[i].indexW(nodeName) >= 0) {
                MFnSet setFn(setNode->thisMObject());
                setFn.addMember(node);
                break;
            }
        }
    }
}

void ExpSet::removeCallbacks() {
    for (unsigned int i = 0; i < m_callbackIds.length(); i++) {
        MMessage::removeCallback(m_callbackIds[i]);
    }
    m_callbackIds.clear();
}