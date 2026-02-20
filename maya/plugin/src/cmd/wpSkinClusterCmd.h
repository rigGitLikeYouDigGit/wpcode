#pragma once
#include <maya/MPxCommand.h>
#include <maya/MArgList.h>
#include <maya/MSyntax.h>

class WpSkinClusterCmd : public MPxCommand {
public:
    WpSkinClusterCmd();
    virtual ~WpSkinClusterCmd();
    
    virtual MStatus doIt(const MArgList& args) override;
    virtual MStatus redoIt() override;
    virtual MStatus undoIt() override;
    virtual bool isUndoable() const override { return true; }
    
    static void* creator();
    static MSyntax newSyntax();
    
    static const char* kCmdName;
};