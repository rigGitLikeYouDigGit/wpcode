

#include "edgeData.h"
#include "../libEigen.h"
#include "vertexData.h"
#include "manifold.h"

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
	
	we ASSUME that a curve will be cut at a certain driver
	*/
	/* update drivers */
	target.driverDatas.clear();
	for (int i = 0; i < driverDatas.size(); i++) {
		SEdgeDriverData& driverData = driverDatas[i];
		if( driverData.uOnEdge < lowU - EPS_F ){ // driver data not included in new range
			continue;
		}
		if (driverData.uOnEdge - EPS_F > highU) { // driver data not included in new range
			continue;
		}
		target.driverDatas.emplace_back(driverData);
		SEdgeDriverData& newData = target.driverDatas.back();
		float newU = remap(driverData.uOnEdge, lowU, highU, 0.0f, 1.0f, true);
		newData.uOnEdge = newU;
	}
	target.subspaceDriver = index;
	target.finalCurve = splitBezPath(finalCurve, lowU, highU);
	target.finalNormals = resampleVectorArray(target.finalNormals, lowU, highU, target.densePointCount());
	return s;
}

void SEdgeData::sortVertices(StrataManifold& manifold) {
	VertexAlongEdgeSorter sorter;
	sorter.manifold = &manifold;
	sorter.alongEdgeId = index;
	std::sort(vertices.begin(), vertices.end(), sorter);
}

aabb::AABB SEdgeData::getAABB() {
	return finalCurve.getAABB();
}