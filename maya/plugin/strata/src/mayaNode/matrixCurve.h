#pragma once

#include "../MInclude.h"

/* strata spinoff:
seems very useful to have the matrix curve available outside a full strata graph
*/

# define MATCURVE_NODE_STATIC_MEMBERS(prefix, nodeT) \
prefix MObject nodeT aMatrixStartIn;\
prefix MObject nodeT aMatrixEndIn;\
prefix MObject nodeT aMatrixMidIn;\
prefix MObject nodeT aMatrixMidInMatrix;\
prefix MObject nodeT aCurveRootResIn;\
prefix MObject nodeT aCurvePointResIn;\
prefix MObject nodeT aSampleIn;\
prefix MObject nodeT aSampleInParam;\
prefix MObject nodeT aSampleOut;\
prefix MObject nodeT aSampleOutMatrix;\
prefix MObject nodeT aCurveOut;\



class MatrixCurveNode : public MPxNode {
public:
	using thisT = MatrixCurveNode;
	MatrixCurveNode() {}
	virtual ~MatrixCurveNode() {}

	static void* creator() {
		MatrixCurveNode* newObj = new MatrixCurveNode();
		return newObj;
	}

	DECLARE_STATIC_NODE_H_MEMBERS(MATCURVE_NODE_STATIC_MEMBERS);

	static MTypeId kNODE_ID;// = const MTypeId(0x00122C1C);
	static MString kNODE_NAME;// = MString("curveFrame");

	static  MString     drawDbClassification;
	static  MString     drawRegistrantId;

	std::vector<MMatrix> cachedMats;

	static MStatus initialize();

	virtual MStatus compute(const MPlug& plug, MDataBlock& data);

	MStatus updateMatrixCache(MDataBlock& data);
	MStatus updateCurve(MDataBlock& data);

};

