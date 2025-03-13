

#include "manifold.h"

using namespace ed;




//inline std::vector<int> SPoint::otherNeighbourPoints(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result;
//	result.reserve(8);
//	for (int edgeId : edges) {
//		StrataElement* edgePtr = manifold.getEl(edgeId);
//		for (int ptId : edgePtr->points) {
//			if (ptId == globalIndex) { continue; }
//			result.push_back(ptId);
//		}
//	}
//	return result;
//}
//inline std::vector<int> SPoint::otherNeighbourEdges(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result;
//	for (int edgeId : edges) {
//		result.push_back(edgeId);
//	}
//	return result;
//}
//
////inline std::vector<int> SPoint::edgeStar(StrataManifold& manifold) {
////	return otherNeighbourEdges(manifold);
////}
//
//inline std::vector<int> SPoint::otherNeighbourFaces(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return all points connected to edges here
//	std::vector<int> result(6, -1);
//	std::set<int> foundFaces;
//	for (int edgeId : edges) {
//		StrataElement* edge = manifold.getEl(edgeId);
//		foundFaces.insert(edge->faces.begin(), edge->faces.end());
//	}
//	for (int faceId : foundFaces) {
//		result.push_back(faceId);
//	}
//	return result;
//}
//
//
//inline std::vector<int> SEdge::otherNeighbourPoints(StrataManifold& manifold) {
//	// return list of pointers to neighbours of the same element type
//	// return end points of this edge only, for now
//	//SmallList<int, 32> result(2);
//	std::vector<int> result;
//
//	if (manifold.getEl(drivers[0])->elType == SElType::point) {
//		result.push_back(drivers[0]);
//	}
//	if (manifold.getEl(drivers[1])->elType == SElType::point) {
//		result.push_back(drivers[1]);
//	}
//	return result;
//}
//
//inline std::vector<int> SEdge::otherNeighbourEdges(StrataManifold& manifold) {
//	std::vector<int> result;
//	result.reserve(8);
//	for (int& ptId : otherNeighbourPoints(manifold)) {
//		for (int& edgeId : ((manifold.getEl(ptId))->otherNeighbourEdges(manifold))) {
//			result.push_back(edgeId);
//		}
//	}
//	auto removeResult = std::remove(
//		result.begin(), result.end(), globalIndex); // remove this edge if it's added
//	return result;
//}


//struct SFace : StrataElement {
//	SElType elType = SElType::face;
//
//	std::vector<bool> edgeOrients; // SURELY vector<bool> is cringe
//	// true for forwards, false for backwards
//	bool flipFace; // if face as a whole should be flipped, after edge winding
//
//	SFace(std::string elName) : StrataElement(elName) {
//	}
//};



