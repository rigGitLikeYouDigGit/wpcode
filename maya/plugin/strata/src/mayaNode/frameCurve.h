#pragma once

#include <vector>
#include <memory>
#include "../MInclude.h"
#include "../lib.h"
#include "../libEigen.h"

#define DEBUGAV(vec) \
for(auto const& i: vec){ \
	COUT << "{" << i[0] <<","<<i[1]<<","<<i[2]<<"}," << " "; \
} COUT << "length " << vec.length() << std::endl;

# define FRAMECURVE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aCurveIn; \
prefix MObject nodeT aRefCurveIn; \
prefix MObject nodeT aSteps; \
\
prefix MObject nodeT aSampleIn; \
prefix MObject nodeT aRefSampleMatrixIn; \
prefix MObject nodeT aActiveSampleMatrixIn; \
prefix MObject nodeT aSampleUIn; \
\
prefix MObject nodeT aSampleOut; \
prefix MObject nodeT aSampleUOut; \
prefix MObject nodeT aSampleMatrixOut; \
prefix MObject nodeT aSampleMatrixOnCurveOut; \
\
prefix MObject nodeT aRefUp; \
prefix MObject nodeT aRefUpMatrix; \
prefix MObject nodeT aRefUpTwist; \
\
prefix MObject nodeT aActiveUp; \
prefix MObject nodeT aActiveUpMatrix; \
prefix MObject nodeT aActiveUpTwist; \

/* if sample matrix in is identity, use U
else use matrix

TODO: LENGTH
on Dragon I did the full kebab for every param of every curve we used - 
blend between u and length
blend between fromStart and fromEnd
blend between normalised and non-normalised

would be nice to have a common way of dealing with that in c++
*/

class FrameCurveNode : public MPxNode {
public:
	DECLARE_STATIC_NODE_H_MEMBERS(FRAMECURVE_STATIC_MEMBERS);

	FrameCurveNode() {}
	virtual ~FrameCurveNode() {}

	static void* creator() {
		FrameCurveNode* newObj = new FrameCurveNode();
		return newObj;
	}


	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;


	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);



	//void postConstructor();

	//MStatus legalConnection(
	//	const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc,
	//	bool& isLegal
	//) const;

	//virtual MStatus connectionMade(const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc
	//);

	//virtual MStatus connectionBroken(const MPlug& plug,
	//	const MPlug& otherPlug,
	//	bool 	asSrc
	//);

};


