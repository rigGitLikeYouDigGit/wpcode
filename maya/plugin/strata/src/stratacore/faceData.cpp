
#include "faceData.h"
#include "manifold.h"
#include "../libEigen.h"

using namespace strata;
std::tuple<int, float, float> SFaceData::edgeSpan(StrataManifold& manifold, int borderIndex) {
	Vertex& startV = manifold.vertices[borderIndex];
	Vertex& endV = manifold.vertices[(borderIndex + 1) % nBorderEdges()];
	
}


std::pair<Vertex*, Vertex*> SFaceData::vtxPair(StrataManifold& man, int borderIndex) {
	return std::make_pair(
		man.getVertex(vertices[borderIndex]),
		man.getVertex(vertices[(borderIndex + 1) % nBorderEdges()])
	);
}

SEdgeData& SFaceData::eDataForBorder(StrataManifold& man, int borderIndex) {
	return man.eDataMap[
		man.getEl(
			man.getVertex(
				vertices[borderIndex]
			)->edgeIds[1]
		)->name
	];
}


float SFaceData::map01UCoordToEdge(StrataManifold& man, float u, int vtxA, int vtxB) {

	Vertex* vA = man.getVertex(vtxA);
	Vertex* vB = man.getVertex(vtxB);
	
	/* check edge direction */
	float uLow = vA->edgeUs[1];
	float uHigh = vB->edgeUs[0];
	if (!vA->edgeDirs[1]) {
		uLow = vB->edgeUs[0];
		uHigh = vA->edgeUs[1];
	}
	/* remap */
	return remap(u, 0.0f, 1.0f, uLow, uHigh);
}

float SFaceData::map01UCoordToEdge(StrataManifold& man, float u, int borderIndex) {
	return map01UCoordToEdge(man, u, vertices[borderIndex], vertices[(borderIndex + 1) % nBorderEdges()]);
}

