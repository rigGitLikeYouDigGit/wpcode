

#include "edgeData.h"
#include "../libEigen.h"

using namespace strata;

Status& strata::SEdgeData::buildFinalBuffers(Status& s) {
	/*
	assumes we have a final curve built in worldspace -
	sample upvectors to normals by dense params

	how do we tell the system what upvectors should be?
	no idea :)
	*/

	MatrixX3f targetNormals(1, 3);
	targetNormals.row(0) = Vector3f{ 0, 1, 0 }; // ABSOLUTE TRASH
	/* I know but consider cases where we just have a literal curve with no drivers,
	where we only have 1, etc
	*/
	finalNormals = makeRMFNormals(
		finalCurve,
		targetNormals,
		densePointCount()
	);


	return s;

}


Status& SEdgeData::getSubData(Status& s, SEdgeData& target, float lowU, float highU) {
	/* copy edge data but curtail final curve to given params
	TODO: inherit normals/upvectors
	*/

	target.finalCurve = splitBezPath(finalCurve, lowU, highU);
	target.finalNormals = resampleVectorArray(target.finalNormals, lowU, highU, target.densePointCount());
	return s;
}

