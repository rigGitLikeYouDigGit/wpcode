
#include "faceData.h"
#include "manifold.h"

using namespace strata;
std::tuple<int, float, float> SFaceData::edgeSpan(StrataManifold& manifold, int borderIndex) {
	Vertex& startV = manifold.vertices[borderIndex];
	Vertex& endV = manifold.vertices[(borderIndex + 1) % nBorderEdges()];
	
}