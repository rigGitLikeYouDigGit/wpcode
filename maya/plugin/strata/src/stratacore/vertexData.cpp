
#include "vertexData.h"
#include "manifold.h"


using namespace strata;

bool VertexAlongEdgeSorter::operator()(const Vertex& left, const Vertex& right) {
	/* no checks done if edge id matches in either vertex */
	float leftVal = left.edgeUs[0];
	float rightVal = right.edgeUs[0];
	if (left.edgeIds[1] == alongEdgeId) {
		leftVal = left.edgeUs[1];
	}
	if (right.edgeIds[1] == alongEdgeId) {
		rightVal = right.edgeUs[1];
	}
	return leftVal < rightVal;
}
bool VertexAlongEdgeSorter::operator()(const int& leftId, const int& rightId) {
	/* no checks done if edge id matches in either vertex */
	Vertex& left = manifold->vertices[leftId];
	Vertex& right = manifold->vertices[rightId];
	return operator()(left, right);
}

//std::array<int, 2> Vertex::edgeIds(StrataManifold& manifold) {
//	return {
//		manifold.hedge(pHedge)->edgeIndex,
//		manifold.hedge(hedge)->edgeIndex
//	};
//}
//std::array<float, 2> Vertex::edgeUs(StrataManifold& manifold) {
//	return {
//		manifold.hedge(pHedge)->us[1],
//		manifold.hedge(hedge)->us[0]
//	};
//}
//std::array<bool, 2> Vertex::edgeDirs(StrataManifold& manifold) {
//	return {
//		manifold.hedge(pHedge)->isFlip(),
//		manifold.hedge(hedge)->isFlip()
//	};
//}
//
//
//std::array<int, 2> HEdge::vtxIds(StrataManifold& manifold) {
//	return {
//	manifold.vtx()->isFlip(),
//	manifold.hedge(hedge)->isFlip()
//	};
//}
//

