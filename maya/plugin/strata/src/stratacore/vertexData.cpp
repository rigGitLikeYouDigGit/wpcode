
#include "vertexData.h"
#include "manifold.h"


using namespace strata;


std::array<int, 2> Vertex::edgeIds(StrataManifold& manifold) {
	return {
		manifold.hedge(pHedge)->edgeIndex,
		manifold.hedge(hedge)->edgeIndex
	};
}
std::array<float, 2> Vertex::edgeUs(StrataManifold& manifold) {
	return {
		manifold.hedge(pHedge)->us[1],
		manifold.hedge(hedge)->us[0]
	};
}
std::array<bool, 2> Vertex::edgeFlips(StrataManifold& manifold) {
	return {
		manifold.hedge(pHedge)->isFlip(),
		manifold.hedge(hedge)->isFlip()
	};
}


std::array<int, 2> HEdge::vtxIds(StrataManifold& manifold) {
	return {
	manifold.vtx()->isFlip(),
	manifold.hedge(hedge)->isFlip()
	};
}


