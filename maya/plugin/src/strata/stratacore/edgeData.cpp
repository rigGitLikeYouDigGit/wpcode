

#include "edgeData.h"
#include "../libEigen.h"
#include "vertexData.h"
#include "manifold.h"

using namespace strata;

Status& strata::SEdge::buildFinalBuffers(Status& s) {
	/*
	assumes we have a final curve built in worldspace -
	sample upvectors to normals by dense params
	*/

	Array3Xf targetNormals(anchorDatas.size(), 3);
	ArrayXf normalParams(anchorDatas.size());
	ArrayXf normalWeights(anchorDatas.size());
	ArrayXf twistValues(anchorDatas.size());
	for(auto i = 0; i < anchorDatas.size(); i++) {
		SEdgeAnchorData& anchorData = anchorDatas[i];
		targetNormals.row(i) = anchorData.normal;
		normalParams(i) = anchorData.uOnEdge;
		normalWeights(i) = 1.0;
		twistValues(i) = anchorData.twist;
	}
	finalNormals = makeRMFNormals(
		finalPositions,
		targetNormals,
		normalParams,
		normalWeights,
		twistValues
	);

	return s;

}


/* REDO subspaces later */
//Status& SEdge::getSubData(Status& s, SEdge& target, float lowU, float highU) {
//	/* copy edge data but curtail final curve to given params
//	
//	we ASSUME that a curve will be cut at a certain anchor
//	*/
//	/* update anchors */
//	target.anchorDatas.clear();
//	for (int i = 0; i < anchorDatas.size(); i++) {
//		SEdgeAnchorData& anchorData = anchorDatas[i];
//		if( anchorData.uOnEdge < lowU - EPS_F ){ // anchor data not included in new range
//			continue;
//		}
//		if (anchorData.uOnEdge - EPS_F > highU) { // anchor data not included in new range
//			continue;
//		}
//		target.anchorDatas.push_back(anchorData);
//		SEdgeAnchorData* newData = target.anchorDatas.back();
//		float newU = remap(anchorData.uOnEdge, lowU, highU, 0.0f, 1.0f, true);
//		newData->uOnEdge = newU;
//	}
//	target.subspaceAnchor = index;
//	target.finalCurve = splitBezPath(finalCurve, lowU, highU);
//	target.finalNormals = resampleVectorArray(target.finalNormals, lowU, highU, target.densePointCount());
//	return s;
//}

void SEdge::sortVertices(StrataManifold& manifold) {
	VertexAlongEdgeSorter sorter;
	sorter.manifold = &manifold;
	sorter.alongEdgeId = elIndex;
	std::sort(vertices.begin(), vertices.end(), sorter);
}

aabb::AABB<3, float> SEdge::getAABB() {
	return strata::getAABB(finalPositions);
}

static SEdge testEdge("testEdge", SElType::edge);
//testEdge.globalIndex = 999;
