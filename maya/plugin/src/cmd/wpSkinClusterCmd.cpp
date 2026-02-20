#include "wpSkinClusterCmd.h"
#include <maya/MArgDatabase.h>
#include <maya/MSelectionList.h>
#include <maya/MGlobal.h>
#include <maya/MDagPath.h>
#include <maya/MFnSkinCluster.h>

const char* WpSkinClusterCmd::kCmdName = "wpSkin";

// Add constructor implementation
WpSkinClusterCmd::WpSkinClusterCmd() 
    : MPxCommand()
{
}

// Add destructor implementation
WpSkinClusterCmd::~WpSkinClusterCmd() {
}

void* WpSkinClusterCmd::creator() {
    return new WpSkinClusterCmd();
}

MSyntax WpSkinClusterCmd::newSyntax() {
    MSyntax syntax;
    
    // Geometry selection
    syntax.setObjectType(MSyntax::kSelectionList, 1, 255);
    syntax.useSelectionAsDefault(true);
    
    // Flags matching skinCluster
    syntax.addFlag("-mi", "-maxInfluences", MSyntax::kLong);
    syntax.addFlag("-dr", "-dropoffRate", MSyntax::kDouble);
    syntax.addFlag("-tsb", "-toSelectedBones", MSyntax::kBoolean);
    
    return syntax;
}

MStatus WpSkinClusterCmd::doIt(const MArgList& args) {
    MStatus status;
    MArgDatabase argData(syntax(), args, &status);
    
    // Get geometry
    MSelectionList selection;
    argData.getObjects(selection);
    
    MDagPath geoPath;
    selection.getDagPath(0, geoPath);
    
    // Create wpSkinCluster
    MString cmd = "deformer -type wpSkinCluster ";
    cmd += geoPath.fullPathName();
    
    MString result;
    status = MGlobal::executeCommand(cmd, result);
    
    // Get influences from remaining selection
    for (unsigned int i = 1; i < selection.length(); ++i) {
        MDagPath infPath;
        selection.getDagPath(i, infPath);
        
        cmd = "skinCluster -e -ai ";
        cmd += infPath.fullPathName();
        cmd += " -lw true -wt 0 ";
        cmd += result;
        
        MGlobal::executeCommand(cmd);
    }
    
    setResult(result);
    return MS::kSuccess;
}

MStatus WpSkinClusterCmd::redoIt() { return MS::kSuccess; }
MStatus WpSkinClusterCmd::undoIt() { return MS::kSuccess; }