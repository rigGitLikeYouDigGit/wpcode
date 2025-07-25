#pragma once


// maya h files are pretty lightweight
// so just include all the main ones

/* MGeometry comes with a warning that you shouldn't use
MGeometryPrimitive -
for once this doesn't seem like a me problem
so we mute all warnings from including maya headers
*/

#pragma warning( push )
#pragma warning( disable : 4996)

#include <maya/MObject.h>
#include <maya/MObjectHandle.h>
#include <maya/MStatus.h>
#include <maya/MMatrix.h>
#include <maya/MString.h>
#include <maya/MPlug.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataBuilder.h>
#include <maya/MArrayDataHandle.h>


#include <maya/MQuaternion.h>
#include <maya/MFloatArray.h>
#include <maya/MDoubleArray.h>
#include <maya/MVectorArray.h>
#include <maya/MPointArray.h>
#include <maya/MPlugArray.h>
#include <maya/MMatrixArray.h>
#include <maya/MStringArray.h>

#include <maya/MMatrix.h>
#include <maya/MDrawRegistry.h>
#include <maya/MUserData.h>
#include <maya/MDrawContext.h>
#include <maya/MHWGeometryUtilities.h>
#include <maya/MGlobal.h>
#include <maya/MEventMessage.h>
#include <maya/MDGModifier.h>

#include <maya/MFnDependencyNode.h>
#include <maya/MFnTransform.h>
#include <maya/MFnNurbsCurve.h>

#include <maya/MFnCompoundAttribute.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixAttribute.h>
#include <maya/MFnGenericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnMessageAttribute.h>

#include <maya/MFnNurbsCurveData.h>
#include <maya/MFnNumericData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnStringData.h>

#include <maya/MEventMessage.h>
#include <maya/MDagMessage.h>
#include <maya/MDGMessage.h>

#include <maya/MFnDependencyNode.h>
#include <maya/MFnTransform.h>

#include <maya/MPxNode.h>
#include <maya/MPxDrawOverride.h>
#include <maya/MPxLocatorNode.h>
#include <maya/MPxSurfaceShape.h>
#include <maya/MPxComponentShape.h>


#include <maya/MPxSurfaceShapeUI.h>
#include <maya/MPxGeometryOverride.h>
#include <maya/MDrawContext.h>
#include <maya/MDrawRegistry.h>
#include <maya/MShaderManager.h>
#include <maya/MSelectionMask.h>

#include <maya/MHWGeometry.h>
#include <maya/MGeometry.h>

#pragma warning( pop ) 
